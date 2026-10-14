"""
adaptive_graph_mix_curmgae.py
==============================
AdaMix + Cur-MGAE 融合版本（experiment-ready：增加消融/鲁棒性/机制诊断开关，默认行为不变）。

在原始 adaptive_graph_mix8.py 的基础上，参考论文：
  "Self-Supervised Masked Graph AutoEncoder via Structure-Aware Curriculum"
  (Cur-MGAE, ICLR 2025 under review)

引入四项关键改进：
  [MOD-1] EdgeDifficultyEstimator   — 边级重建残差难度评分，替换 spot 级 EMA-MSE
  [MOD-2] CrossCorrelationDecoder  — 多层嵌入逐元素积解码器，用于边存在概率预测
  [MOD-3] CurriculumEdgeMasker     — 自步 (self-paced) 边掩码调度器，实现
                                     "easy → hard" 边课程遮蔽策略
  [MOD-4] PrototypeContrastiveLoss — 可导原型对比聚类损失（DEC 自训练 + 均匀性正则），
                                     直接在 latent space 上施加簇内紧凑/簇间分离约束

原始代码中保留不变的部分：
  * ImageFeatureExtractor（H&E 图像特征提取）
  * AdaptiveGraphBuilder （跨模态边构建）
  * SelfPacedCurriculum  （age_param / get_weight，继续用于图强度/扰动分数）
  * AdaMixGraphST 初始化流程
  * GraphST Encoder 架构（不修改原始 GNN）

修改摘要（在代码中以 "[MOD-X]" 标注）：
  - 新增 EdgeDifficultyEstimator（约 120 行）
  - 新增 CrossCorrelationDecoder（约 60 行）
  - 新增 CurriculumEdgeMasker（约 130 行）
  - 新增 PrototypeContrastiveLoss（约 110 行）[MOD-4]
  - AdaMixGraphST.__init__ 中替换 confidence_estimator → edge_difficulty_estimator，
    新增 edge_masker / cc_decoder / proto_loss_fn 字段
  - AdaMixGraphST.train() 循环中：
      * 用边难度替代 spot 置信度驱动 curriculum
      * 调用 CrossCorrelationDecoder 计算边重建损失
      * 调用 CurriculumEdgeMasker 选择本轮遮蔽边集合
      * 调用 PrototypeContrastiveLoss 计算聚类损失（warmup 后激活）
"""

import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import h5py

# =============================================================================
# [MOD-1]  EdgeDifficultyEstimator
#          替换原始 SpotConfidenceEstimator
#          核心思想来自 Cur-MGAE §3.2：
#            R = A − A_re  （残差越小 = 边越容易重建）
# =============================================================================

class GraphTopologyRegularization(nn.Module):
    def __init__(self, align_weight=1.0, sparse_weight=0.1):
        super().__init__()
        self.align_weight = align_weight
        self.sparse_weight = sparse_weight

    def forward(self, adj_spatial, adj_image):
        if adj_spatial.is_sparse:
            adj_spatial = adj_spatial.to_dense()
        if adj_image.is_sparse:
            adj_image = adj_image.to_dense()

        loss_align = F.mse_loss(adj_spatial, adj_image)
        
        loss_sparse = torch.mean(torch.abs(adj_spatial))
        
        total_reg_loss = (self.align_weight * loss_align) + (self.sparse_weight * loss_sparse)
        return total_reg_loss

class EdgeDifficultyEstimator:
    """
    per-edge 重建难度估计器。

    Cur-MGAE 中：
        difficult(e_{ij}) ∝ ||A_{ij} - Ã_{ij}||   （重建残差）

    空间转录组适配：
        - 边 e_{ij} 对应空间邻近 spot 对 (i, j)
        - "A_{ij}" 用两 spot 嵌入的余弦相似度代理 (真实标签: 1.0)
        - "Ã_{ij}" 为模型预测的边存在概率（来自 CrossCorrelationDecoder）
        - 残差 R_{ij} = |1.0 − sigmoid(logit_{ij})|

    Parameters
    ----------
    edge_index : (2, E)  int64  — COO 格式的有向/无向边索引
    ema_decay  : EMA 平滑系数（越高 = 更新越慢）
    """

    def __init__(self, edge_index: np.ndarray, ema_decay: float = 0.7):
        self.edge_index = edge_index          # (2, E)
        self.n_edges    = edge_index.shape[1]
        self.ema_decay  = ema_decay

        # 初始化：所有边难度相同（中等）
        self.ema_residual = np.full(self.n_edges, 0.5, dtype=np.float32)
        self._initialized = False

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, edge_logits: torch.Tensor,
               edge_indices: Optional[np.ndarray] = None) -> None:
        """
        用解码器输出更新边难度 EMA。

        Parameters
        ----------
        edge_logits  : (K,)  — CrossCorrelationDecoder 对被遮蔽边的 logit 输出
        edge_indices : (K,)  — 对应的边局部索引（在 edge_index 中的列号）
                               若为 None 则假定覆盖全部边（向后兼容）
        """
        probs     = torch.sigmoid(edge_logits).cpu().numpy().astype(np.float32)
        residuals = np.abs(1.0 - probs)                # 真实边标签为 1

        self._initialized = True   # ema_residual 始终保持 (n_edges,) 形状

        if edge_indices is None:
            # 覆盖全部边（仅当 logits 长度等于 n_edges 时使用）
            if len(residuals) == self.n_edges:
                self.ema_residual = (
                    self.ema_decay * self.ema_residual
                    + (1.0 - self.ema_decay) * residuals
                )
        else:
            # 只更新被遮蔽的 K 条边，其余保持不变
            idx = np.asarray(edge_indices).ravel().astype(np.int64)
            valid = (idx >= 0) & (idx < self.n_edges)
            idx_v = idx[valid]
            res_v = residuals[valid]
            self.ema_residual[idx_v] = (
                self.ema_decay * self.ema_residual[idx_v]
                + (1.0 - self.ema_decay) * res_v
            )

    # ------------------------------------------------------------------
    def get_difficulty(self) -> np.ndarray:
        """
        返回 per-edge 难度分数 ∈ [0,1]（越高 = 越难重建）。
        """
        mn, mx = self.ema_residual.min(), self.ema_residual.max()
        return ((self.ema_residual - mn) / (mx - mn + 1e-8)).astype(np.float32)

    # ------------------------------------------------------------------
    def get_spot_difficulty(self, n_spots: int) -> np.ndarray:
        diff   = self.get_difficulty()                        # (E,)
        agg    = np.zeros(n_spots, dtype=np.float32)
        counts = np.zeros(n_spots, dtype=np.float32)

        src = np.asarray(self.edge_index[0]).ravel().astype(np.int64)
        dst = np.asarray(self.edge_index[1]).ravel().astype(np.int64)
        valid = (src >= 0) & (src < n_spots) & (dst >= 0) & (dst < n_spots)
        src, dst, diff_v = src[valid], dst[valid], diff[valid]

        np.add.at(agg,    src, diff_v)
        np.add.at(counts, src, 1.0)
        np.add.at(agg,    dst, diff_v)
        np.add.at(counts, dst, 1.0)

        counts = np.maximum(counts, 1.0)
        return (agg / counts).astype(np.float32)

    # ------------------------------------------------------------------
    def get_spot_confidence(self, n_spots: int) -> np.ndarray:
        """
        spot 级置信度（1 - difficulty），供原始 SelfPacedCurriculum 使用。
        """
        return 1.0 - self.get_spot_difficulty(n_spots)


# =============================================================================
# [MOD-2]  CrossCorrelationDecoder
#          对应 Cur-MGAE §3.1 的跨相关解码器
#          h_{e_{v,u}} = || ^K_{k,j=1}  h_v^(k) ⊙ h_u^(j)
#          g(v,u) = MLP(h_{e_{v,u}})
# =============================================================================

class CrossCorrelationDecoder(nn.Module):
    """
    多层嵌入逐元素积解码器，预测每对节点（spot）之间的边存在概率。

    与原始内积解码器的区别：
        - 逐元素乘法保留共享特征，抑制差异分量
        - 使用多层（K 层）嵌入的拼接，捕捉多粒度结构信息
        - 最终 MLP 层输出单 logit（配合 BCEWithLogitsLoss）

    Parameters
    ----------
    embed_dim : 每层嵌入的维度（等于 GNN encoder 的 dim_output）
    n_layers  : 参与编码的 GNN 层数（GraphST Encoder 输出 1 层，K=1）
    hidden_dim: MLP 中间层宽度
    """

    def __init__(self, embed_dim: int, n_layers: int = 1, hidden_dim: int = 128):
        super().__init__()
        # 输入维度 = n_layers^2 * embed_dim（笛卡尔积的拼接）
        in_dim = (n_layers ** 2) * embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        h_list: List[torch.Tensor],   # 每层嵌入，长度 K，每个 (N, d)
        edge_index: torch.Tensor,     # (2, E)  int64
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : (E,)  — 每条边的存在 logit（未经 sigmoid）
        """
        src, dst = edge_index[0], edge_index[1]   # (E,)

        parts = []
        for hk in h_list:
            for hj in h_list:
                # 逐元素积：(E, d)
                parts.append(hk[src] * hj[dst])

        # 拼接所有粒度组合：(E, K^2 * d)
        h_edge = torch.cat(parts, dim=-1)

        logits = self.mlp(h_edge).squeeze(-1)     # (E,)
        return logits


# =============================================================================
# [MOD-4]  PrototypeContrastiveLoss
#          可导的无监督聚类损失，直接作用于 latent embedding
#          设计目标：在不使用真实标签的前提下，让高维 latent space
#          中同类 spot 尽量聚集、不同类 spot 尽量分离
#
#          理论基础：
#            - DEC (Xie et al., 2016) 的软分配自训练目标（KL 散度损失）
#            - SwAV (Caron et al., 2020) 的原型均匀性约束（熵正则）
#            - 空间平滑正则：相邻 spot 的软分配分布应相似（KL 散度）
#
#          为什么比直接在 latent 上聚类效果好：
#            - 将"聚类目标"显式加入训练损失，弥补重构目标与聚类目标的 gap
#            - 原型作为可学习参数，与 encoder 协同优化
#            - 均匀性约束防止所有 spot 塌缩到同一原型（模式崩塌）
#            - 空间平滑正则利用空间先验，与 ST 数据的层级结构先验一致
# =============================================================================

class PrototypeContrastiveLoss(nn.Module):
    """
    可导的原型对比聚类损失。

    损失由三部分组成：
      (1) L_compact  — DEC 自训练 KL 散度：让 soft assignment q 向 sharp 目标 p 靠拢
      (2) L_uniform  — 均匀性熵正则：防止所有 spot 塌缩到同一个原型
      (3) L_spatial  — 空间平滑正则（可选）：相邻 spot 的分配分布应相似

    Parameters
    ----------
    n_clusters  : 预期聚类数（细胞类型数）
    embed_dim   : latent embedding 的维度（等于 dim_output）
    temperature : soft assignment 的温度参数（越低 = 分配越尖锐）
    uniform_weight : 均匀性正则权重
    spatial_weight : 空间平滑正则权重（0 = 不启用）
    """

    def __init__(
        self,
        n_clusters: int,
        embed_dim: int,
        temperature: float = 0.2,
        uniform_weight: float = 0.1,
        spatial_weight: float = 0.5,
    ):
        super().__init__()
        self.n_clusters     = n_clusters
        self.temperature    = temperature
        self.uniform_weight = uniform_weight
        self.spatial_weight = spatial_weight
        self.initialized = False

        # K 个可学习原型，L2 归一化后使用
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(n_clusters, embed_dim), dim=1)
        )

    @torch.no_grad()
    def initialize_prototypes(self, z: torch.Tensor):

        z_np = F.normalize(z, dim=1).detach().cpu().numpy()

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            random_state=0,
        )

        kmeans.fit(z_np)

        centers = torch.tensor(
            kmeans.cluster_centers_,
            dtype=z.dtype,
            device=z.device,
        )

        centers = F.normalize(centers, dim=1)

        self.prototypes.data.copy_(centers)

        self.initialized = True

        print("[PrototypeContrastiveLoss] prototypes initialized by KMeans.")

    # ------------------------------------------------------------------
    def _soft_assignment(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算 soft assignment 矩阵 q ∈ (N, K)。
        使用余弦相似度 + softmax（温度缩放）。
        """
        z_norm    = F.normalize(z, dim=1)                          # (N, d)
        proto     = F.normalize(self.prototypes, dim=1)             # (K, d)
        logits    = z_norm @ proto.T / self.temperature             # (N, K)
        return F.softmax(logits, dim=1)                             # (N, K)

    # ------------------------------------------------------------------
    @staticmethod
    def _sharpen(q: torch.Tensor) -> torch.Tensor:
        """
        目标分布 p（DEC 论文公式）：
          p_ij = (q_ij^2 / f_j) / sum_j'(q_ij'^2 / f_j')
          其中 f_j = sum_i q_ij（软聚类频率）
        尖锐化使 p 比 q 更靠近 one-hot，提供更强的自训练信号。
        """
        q2   = q ** 2
        f    = q2.sum(dim=0, keepdim=True)                          # (1, K)
        p    = q2 / (f + 1e-8)
        return p / (p.sum(dim=1, keepdim=True) + 1e-8)             # (N, K)

    # ------------------------------------------------------------------
    def forward(
        self,
        z: torch.Tensor,
        spatial_nbr_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z               : (N, d)  latent embedding（hiden_feat）
        spatial_nbr_idx : (N, k)  int64，每个 spot 的 k 个空间邻居索引
                          若为 None 则跳过空间平滑正则

        Returns
        -------
        loss : scalar
        """
        q = self._soft_assignment(z)                                # (N, K)

        # ---- (1) DEC 自训练 KL 散度损失 ----
        with torch.no_grad():
            p = self._sharpen(q)                                    # (N, K)
        # KL(p || q)，reduction='batchmean' 已除以 N
        loss_compact = F.kl_div(
            (q + 1e-8).log(), p, reduction='batchmean'
        )

        # ---- (2) 均匀性正则（最大化原型分配熵）----
        # 鼓励每个原型被大致等量的 spot 分配到，防止模式崩塌
        mean_q       = q.mean(dim=0)                                # (K,)
        loss_uniform = (mean_q * (mean_q + 1e-8).log()).sum()       # 负熵（最小化 = 最大化熵）

        loss = loss_compact + self.uniform_weight * loss_uniform

        # ---- (3) 空间平滑正则（可选）----
        if spatial_nbr_idx is not None and self.spatial_weight > 0.0:
            # spatial_nbr_idx: (N, k) — 每个 spot 的空间邻居局部索引
            N, k   = spatial_nbr_idx.shape
            nbr_q  = q[spatial_nbr_idx.reshape(-1)].reshape(N, k, self.n_clusters)
            # 邻居软分配的均值作为目标
            nbr_mean = nbr_q.mean(dim=1)                            # (N, K)
            nbr_mean = (nbr_mean / (nbr_mean.sum(dim=1, keepdim=True) + 1e-8))
            # KL(nbr_mean || q)：让 spot 的分配与邻居一致
            loss_spatial = F.kl_div(
                (q + 1e-8).log(), nbr_mean.detach(), reduction='batchmean'
            )
            loss = loss + self.spatial_weight * loss_spatial

        return loss


# =============================================================================
# [MOD-3]  CurriculumEdgeMasker
#          对应 Cur-MGAE §3.2 + §3.3
#          管理"先遮蔽容易边、逐步纳入难边"的自步调度
# =============================================================================

class CurriculumEdgeMasker:
    """
    自步边掩码调度器。

    调度逻辑（对应 Cur-MGAE Algorithm 1 核心步骤）：
        1. 根据 EdgeDifficultyEstimator 获取当前边难度
        2. 按难度排序，取前 K 条最容易的边作为"课程边"
        3. split_ratio 控制"按难度精选"与"随机选取"的比例
           - split_ratio = 1.0 → 完全基于难度（类似 Cur-MGAE 的 split_ratio=1）
           - split_ratio = 0.0 → 完全随机（消融基线）
        4. 通过 mask_ratio 上界限制最大遮蔽比例

    Parameters
    ----------
    n_edges     : 总边数
    mask_ratio  : 最大遮蔽比例（对应论文中 ω）
    split_ratio : 难度引导 vs 随机选取的比例（对应论文中 split ratio）
    """

    def __init__(
        self,
        n_edges: int,
        mask_ratio: float = 0.5,
        split_ratio: float = 0.5,
    ):
        self.n_edges    = n_edges
        self.mask_ratio = mask_ratio
        self.split_ratio = split_ratio

        # λ 参数控制本轮应选多少条边（随训练递增）
        self._lambda = 0.0

    # ------------------------------------------------------------------
    def update_lambda(self, epoch: int, total_epochs: int) -> None:
        """
        按训练进度递增 λ（对应 Cur-MGAE 公式 7）：
        前 2/3 epochs 线性增大，之后保持最大值。
        """
        warmup_end = int(total_epochs * (2 / 3))
        if epoch < warmup_end:
            self._lambda = epoch / max(warmup_end, 1)
        else:
            self._lambda = 1.0

    # ------------------------------------------------------------------
    def select_mask_edges(
        self,
        difficulty: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        选择本轮要遮蔽的边索引。

        返回按"先易后难"顺序选出的边局部索引（相对 edge_index 列）。

        Parameters
        ----------
        difficulty : (E,)  边难度分数
        rng        : numpy 随机生成器

        Returns
        -------
        selected : (K_selected,)  int — 被选中遮蔽的边索引
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # K = λ * mask_ratio * E（随训练递增）
        K_target = int(np.ceil(self._lambda * self.mask_ratio * self.n_edges))
        K_target = min(K_target, int(self.mask_ratio * self.n_edges))
        K_target = max(K_target, 1)

        # 按难度升序（最容易的边排在前面）
        order_easy_first = np.argsort(difficulty)   # ascending

        # 难度引导选取（容易优先）
        n_guided = max(1, int(np.ceil(self.split_ratio * K_target)))
        guided_idx = order_easy_first[:n_guided]

        # 随机补充剩余名额
        n_random = K_target - n_guided
        if n_random > 0:
            remaining = np.setdiff1d(
                np.arange(self.n_edges), guided_idx, assume_unique=True
            )
            if len(remaining) > 0:
                n_random = min(n_random, len(remaining))
                random_idx = rng.choice(remaining, size=n_random, replace=False)
                selected = np.concatenate([guided_idx, random_idx])
            else:
                selected = guided_idx
        else:
            selected = guided_idx

        return selected.astype(np.int64)


# =============================================================================
# 保留原始辅助类（未修改）
# =============================================================================

class SpotConfidenceEstimator:
    """原始 SpotConfidenceEstimator，保留供兼容调用（现由 EdgeDifficultyEstimator
    的 get_spot_confidence 替代核心功能）。"""

    def __init__(self, n_spots: int, ema_decay: float = 0.7,
                 use_pearson: bool = True):
        self.n_spots    = n_spots
        self.ema_decay  = ema_decay
        self.use_pearson = use_pearson
        self.ema_mse     = np.ones(n_spots, dtype=np.float32)
        self.ema_pearson = np.zeros(n_spots, dtype=np.float32)
        self.fused_confidence = np.full(n_spots, 0.5, dtype=np.float32)
        self._initialized = False

    def update(self, feat: torch.Tensor, recon: torch.Tensor) -> None:
        with torch.no_grad():
            per_spot_mse = (feat - recon).pow(2).mean(dim=1).cpu().numpy()
            if not self._initialized:
                self.ema_mse = per_spot_mse.copy()
                self._initialized = True
            else:
                self.ema_mse = (self.ema_decay * self.ema_mse
                                + (1.0 - self.ema_decay) * per_spot_mse)
            if self.use_pearson:
                f_np = feat.cpu().numpy().astype(np.float64)
                r_np = recon.cpu().numpy().astype(np.float64)
                pearson = np.array([
                    float(np.corrcoef(f_np[i], r_np[i])[0, 1])
                    if f_np[i].std() > 1e-8 and r_np[i].std() > 1e-8 else 0.0
                    for i in range(self.n_spots)
                ], dtype=np.float32)
                self.ema_pearson = (self.ema_decay * self.ema_pearson
                                    + (1.0 - self.ema_decay) * pearson)
            self.fused_confidence = self._compute_reconstruction_confidence()

    def _compute_reconstruction_confidence(self) -> np.ndarray:
        mse_min, mse_max = self.ema_mse.min(), self.ema_mse.max()
        denom = mse_max - mse_min + 1e-8
        conf  = 1.0 - (self.ema_mse - mse_min) / denom
        if self.use_pearson and self._initialized:
            pearson_norm = (self.ema_pearson + 1.0) / 2.0
            conf = 0.7 * conf + 0.3 * pearson_norm
        return conf.astype(np.float32)

    def get_confidence(self) -> np.ndarray:
        return self.fused_confidence.astype(np.float32)

    def update_with_spatial_consistency(
        self, hiden_feat, spatial_nbr_idx, consistency_weight=0.3
    ):
        with torch.no_grad():
            z = F.normalize(hiden_feat.float(), dim=-1).cpu()
            k = spatial_nbr_idx.shape[1]
            N = z.shape[0]
            nbr_feats = z[spatial_nbr_idx.reshape(-1)].reshape(N, k, -1)
            nbr_mean  = F.normalize(nbr_feats.mean(dim=1), dim=-1)
            cos_sim   = (z * nbr_mean).sum(dim=1).numpy()
            conf_nbr  = (cos_sim.clip(-1, 1) + 1.0) / 2.0
        conf_recon = self._compute_reconstruction_confidence()
        self.fused_confidence = (
            (1.0 - consistency_weight) * conf_recon
            + consistency_weight * conf_nbr.astype(np.float32)
        ).astype(np.float32)

    def split_spots(self, top_fraction=0.33):
        conf = self.get_confidence()
        order = np.argsort(conf)
        n_group = max(1, int(self.n_spots * top_fraction))
        low_conf_idx  = order[:n_group]
        high_conf_idx = order[-n_group:]
        return high_conf_idx, low_conf_idx


# [UNI-REMOVED] ImageFeatureExtractor class deleted (H&E image feature extractor)


class SelfPacedCurriculum:
    """原始 SelfPacedCurriculum，保留用于图强度/扰动控制。"""

    def __init__(self, max_iter: int, K_max: int = 5):
        self.max_iter = max_iter
        self.K_max    = K_max

    def age_param(self, t: int) -> float:
        start, end = 0.20, 0.36
        return start + (end - start) * (t / max(self.max_iter - 1, 1))

    def compute_proxy_loss(self, confidence: np.ndarray) -> float:
        return float(1.0 - confidence.mean())

    def get_mask(self, L_bar: float, lam: float) -> int:
        return 1 if self.get_weight(L_bar, lam) > 0.0 else 0

    def get_weight(self, L_bar: float, lam: float) -> float:
        margin, delta = 0.05, 0.25
        raw_weight = (lam + margin - L_bar) / max(delta, 1e-8)
        return float(np.clip(raw_weight, 0.0, 1.0))

    def n_perturbations(self, v: float) -> int:
        if v <= 0.0:
            return 0
        return max(1, int(np.ceil(v * self.K_max)))


class AdaptiveGraphBuilder:

    def __init__(self, spatial_coords, n_spatial_neighbors=6,
                 n_image_neighbors=5, device=None, top_fraction=0.33):
        # [UNI-REMOVED] n_image_neighbors 保留在签名中仅为兼容旧调用，不再使用
        self.n_spots             = spatial_coords.shape[0]
        self.n_spatial_neighbors = n_spatial_neighbors
        self.device              = device or torch.device('cpu')
        self.spatial_adj         = self._build_knn_adj(spatial_coords,
                                                        n_spatial_neighbors)
        self.image_adj           = None      # [UNI-REMOVED] 永远为 None
        self._spatial_neighbors  = self._adj_to_neighbor_list(self.spatial_adj)
        self._image_neighbors    = None      # [UNI-REMOVED] 永远为 None
        self.coords = np.asarray(spatial_coords, dtype=np.float32)
        self.top_fraction = top_fraction

    @staticmethod
    def _build_knn_adj(coords, k):
        n = coords.shape[0]
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(coords)
        _, indices = nbrs.kneighbors(coords)
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in indices[i, 1:]:
                adj[i, j] = 1.0
        return np.clip(adj + adj.T, 0, 1)
    
    def _local_conf_flags(self, confidence, frac=None):
        if frac is None:
            frac = self.top_fraction
        conf = np.asarray(confidence, dtype=np.float32)
        is_low = np.zeros(self.n_spots, dtype=bool)
        is_high = np.zeros(self.n_spots, dtype=bool)

        for i in range(self.n_spots):
            nbrs = self._spatial_neighbors[i]
            if len(nbrs) == 0:
                continue

            local_idx = np.concatenate([[i], nbrs])
            local_conf = conf[local_idx]

            q_low = np.quantile(local_conf, frac)
            q_high = np.quantile(local_conf, 1.0 - frac)

            is_low[i] = conf[i] <= q_low
            is_high[i] = conf[i] >= q_high

        return is_low, is_high

    @staticmethod
    def _adj_to_neighbor_list(adj):
        return [np.where(adj[i] > 0)[0] for i in range(adj.shape[0])]

    # [UNI-REMOVED] set_image_features() 已删除（不再构建图像 KNN 图）

    def build_adj(self, confidence, mask_m, n_extra, max_img_dist=None):
        # [UNI-REMOVED] image_adj 恒为 None，此处直接返回纯空间图
        return self.spatial_adj.copy()

    def build_perturbed_features(
        self,
        features,
        confidence,
        mask_m,
        n_swaps,
        mix_ratio_easy=0.15,
        mix_ratio_hard=0.30,
        perturb_fraction=0.25,
    ):
        feat_a = features.copy()
        conf = np.asarray(confidence, dtype=np.float32)
        is_low, is_high = self._local_conf_flags(conf, frac=0.33)
        rng = np.random.default_rng(42)

        if mask_m == 0:
            target_spots = np.where(is_low)[0]
            gamma = mix_ratio_easy
        else:
            target_spots = np.where(is_high)[0]
            gamma = mix_ratio_hard

        if len(target_spots) == 0:
            return feat_a

        n_targets = min(
            len(target_spots),
            max(1, int(np.ceil(perturb_fraction * len(target_spots))))
        )
        if n_targets < len(target_spots):
            target_spots = rng.choice(target_spots, size=n_targets, replace=False)

        for i in target_spots:
            spatial_nbrs = set(self._spatial_neighbors[i].tolist())
            # [UNI-REMOVED] 无图像邻居时回退到纯空间邻居（原为 spatial ∩ image 交集）
            if self._image_neighbors is not None:
                img_nbrs = set(self._image_neighbors[i])
                candidates = list(spatial_nbrs.intersection(img_nbrs))
            else:
                candidates = list(spatial_nbrs)

            if mask_m == 0:
                candidates = [j for j in candidates if is_high[j]]
            else:
                candidates = [j for j in candidates if is_low[j]]

            if len(candidates) == 0:
                continue

            chosen = rng.choice(candidates)
            feat_a[i] = (1.0 - gamma) * features[i] + gamma * features[chosen]

        return feat_a


# =============================================================================
# Adjacency preprocessing utilities（未修改）
# =============================================================================

def normalize_adj(adj: np.ndarray) -> np.ndarray:
    import scipy.sparse as sp
    A = sp.coo_matrix(adj)
    rowsum = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-8, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    A_norm = D.dot(A).dot(D)
    return A_norm.toarray() + np.eye(adj.shape[0])


def adj_to_edge_index(adj) -> np.ndarray:
    """
    将二值邻接矩阵转为 COO 格式 edge_index (2, E)。
    兼容 scipy 稀疏矩阵（GraphST 可能以 csr/csc 格式存储 adj）。
    """
    import scipy.sparse as sp
    if sp.issparse(adj):
        coo = adj.tocoo()
        rows = coo.row.astype(np.int64)
        cols = coo.col.astype(np.int64)
        # 只保留正值边
        mask = np.asarray(coo.data).ravel() > 0
        rows, cols = rows[mask], cols[mask]
    else:
        adj_arr = np.asarray(adj)
        rows, cols = np.where(adj_arr > 0)
        rows = rows.astype(np.int64)
        cols = cols.astype(np.int64)
    return np.stack([rows, cols], axis=0)


# =============================================================================
# 预计算图像嵌入加载工具（UNI / 任意预提取特征）
# =============================================================================

def load_img_embs_h5(
    h5_path: str,
    barcode_order=None,
) -> np.ndarray:
    """
    从 img_embs.h5 加载预计算的图像嵌入。

    Parameters
    ----------
    h5_path      : img_embs.h5 文件路径（通常为 <sample_dir>/img_embs.h5）
    barcode_order: 与 adata.obs_names 对齐的 barcode 列表。
                   若提供，则按该顺序对嵌入重排，确保与 spot 一一对应；
                   若为 None，则直接按文件内顺序返回（需调用者自行保证顺序正确）。

    Returns
    -------
    embeddings : (N, D) float32 ndarray，N = spot 数，D = 嵌入维度（如 UNI 为 1024）

    典型用法
    --------
    >>> embs = load_img_embs_h5(
    ...     os.path.join(file_fold, 'img_embs.h5'),
    ...     barcode_order=list(adata.obs_names),
    ... )
    >>> model = AdaMixGraphST(adata, image_embeddings=embs, ...)
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"[load_img_embs_h5] img_embs.h5 not found: {h5_path}\n"
            "  请先运行特征提取脚本 extract_uni_features.py 生成该文件。"
        )

    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:]                              # (N, D)
        raw_barcodes = f["barcodes"][:]
        # 兼容 bytes / str 两种存储格式
        barcodes = [
            b.decode("utf-8") if isinstance(b, (bytes, np.bytes_)) else str(b)
            for b in raw_barcodes
        ]

    embeddings = embeddings.astype(np.float32)

    if barcode_order is not None:
        bc2idx = {bc: i for i, bc in enumerate(barcodes)}
        missing = [bc for bc in barcode_order if bc not in bc2idx]
        if missing:
            raise KeyError(
                f"[load_img_embs_h5] {len(missing)} barcodes from adata not found "
                f"in img_embs.h5 (e.g. {missing[:3]}). "
                "请确认提取特征时使用了相同的样本数据。"
            )
        order      = [bc2idx[bc] for bc in barcode_order]
        embeddings = embeddings[order]

    print(
        f"[load_img_embs_h5] Loaded from {os.path.basename(h5_path)}: "
        f"shape={embeddings.shape}, dtype={embeddings.dtype}"
    )
    return embeddings


# =============================================================================
# AdaMixGraphST — 融合 Cur-MGAE 改进的主类
# =============================================================================

class AdaMixGraphST:
    """
    融合了 Cur-MGAE 三项改进的空间转录组训练框架。

    相比原始 adaptive_graph_mix8.py 的变化：
      [MOD-1] 新增 EdgeDifficultyEstimator，
              提供 per-edge 难度（残差驱动）
      [MOD-2] 新增 CrossCorrelationDecoder，
              以边重建损失（BCE）替代/增强特征重建（MSE）
      [MOD-3] 新增 CurriculumEdgeMasker，
              在训练图中按"先易后难"顺序遮蔽边

    其余参数与 API 与原始版本完全兼容。

    额外参数
    --------
    image_embeddings : (N, D) float32 ndarray，**推荐方式**。
                       直接传入预计算的图像嵌入（如 UNI 1024-dim），
                       从对应样本目录下的 img_embs.h5 中读取即可：
                           embs = load_img_embs_h5(
                               os.path.join(file_fold, 'img_embs.h5'),
                               barcode_order=list(adata.obs_names),
                           )
                       若同时传入 image_patches，image_embeddings 优先。
    image_patches    : (N, 3, H, W) uint8 ndarray，**旧接口（仍支持）**。
                       原始 H&E patch，将在内部通过 ImageFeatureExtractor 提取特征。
                       推荐改用 image_embeddings 以避免重复提取。
    mask_ratio  : [MOD-3] 最大边遮蔽比例（对应 Cur-MGAE ω）
    split_ratio : [MOD-3] 难度引导 vs 随机选取比例（对应论文 split ratio）
    cc_hidden   : [MOD-2] CrossCorrelationDecoder MLP 隐藏层宽度
    gamma_edge  : 边重建损失在总损失中的权重
    n_clusters  : [MOD-4] 预期聚类数（细胞类型数），建议与下游 mclust 保持一致
    gamma_cluster : [MOD-4] 聚类损失权重（建议 0.3~1.0，默认 0.5）
    proto_temperature : [MOD-4] 软分配温度（越低 = 分配越尖锐，默认 0.2）
    proto_uniform_weight : [MOD-4] 均匀性正则权重（默认 0.1）
    proto_spatial_weight : [MOD-4] 空间平滑正则权重（默认 0.5，0 = 不启用）
    """

    def __init__(
        self,
        adata,
        image_patches: Optional[np.ndarray] = None,      # [UNI-REMOVED] deprecated, ignored
        image_embeddings: Optional[np.ndarray] = None,   # [UNI-REMOVED] deprecated, ignored
        epochs: int = 600,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        dim_output: int = 64,
        alpha: float = 10.0,
        beta: float = 1.0,
        K_max: int = 5,
        ema_decay: float = 0.7,
        n_spatial_neighbors: int = 6,
        n_image_neighbors: int = 5,
        use_pretrained_cnn: bool = True,
        top_fraction: float = 0.33,
        max_img_dist: Optional[float] = None,
        random_seed: int = 41,
        consistency_weight: float = 0.3,
        warmup_epochs: int = 100,
        mask_ratio: float = 0.5,
        split_ratio: float = 0.5,
        cc_hidden: int = 128,
        gamma_edge: float = 1.0,
        n_clusters: int = 7,
        gamma_cluster: float = 0.0,  # [EXP] obsolete in current paper configuration; keep OFF
        proto_temperature: float = 0.2,
        proto_uniform_weight: float = 0.1,
        proto_spatial_weight: float = 0.5,
        dim_proto = 3000,
        # ----- [AHSG] latent semantic neighbors (motivated by SpaCross) -----
        use_latent_semantic: bool = True,
        latent_sem_k: int = 6,                 # 每个 spot 的 latent 语义邻居数
        latent_sem_refresh: int = 50,          # 每多少 epoch 重算一次 latent KNN
        latent_sem_strength: float = 0.30,     # 语义图占融合上限
        latent_sem_start_epoch: Optional[int] = None,   # 开始用 latent KNN 的 epoch；None=warmup_epochs
        # ----- [LAP] spatial Laplacian smoothness regularization -----
        gamma_smooth: float = 0.0,             # 拉普拉斯平滑损失权重（0=关闭）
        smooth_start_epoch: int = 0,           # 从第几个 epoch 开始施加平滑
        # ----- [EXP] experiment / ablation controls -----
        curriculum_mode: str = "easy_to_hard", # easy_to_hard | random | hard_to_easy | static_easy | none
        record_diagnostics: bool = True,        # 记录训练动力学，供机制实验作图
        respect_input_adj: bool = False,        # True: graph_builder 使用 adata.obsm['adj']（用于受控图扰动实验）
        use_feature_perturbation: bool = True,  # 机制消融时可关闭，隔离 edge curriculum 的贡献
    ):
        self.adata              = adata.copy()
        self.epochs             = epochs
        self.device             = device or torch.device('cpu')
        self.learning_rate      = learning_rate
        self.weight_decay       = weight_decay
        self.dim_output         = dim_output
        self.alpha              = alpha
        self.beta               = beta
        self.top_fraction       = top_fraction
        self.consistency_weight = consistency_weight
        self.warmup_epochs      = warmup_epochs
        self.gamma_edge         = gamma_edge
        self.gamma_cluster      = gamma_cluster
        self._n_clusters        = n_clusters
        self._proto_temperature = proto_temperature
        self._proto_uniform_w   = proto_uniform_weight
        self._proto_spatial_w   = proto_spatial_weight
        self.max_img_dist = max_img_dist
        self.dim_proto = dim_proto

        # [AHSG] latent semantic neighbors config
        self.use_latent_semantic   = use_latent_semantic
        self.latent_sem_k          = latent_sem_k
        self.latent_sem_refresh    = latent_sem_refresh
        self.latent_sem_strength   = latent_sem_strength
        self.latent_sem_start_epoch = (
            latent_sem_start_epoch if latent_sem_start_epoch is not None
            else warmup_epochs
        )
        # 缓存：当前 latent semantic adj（未归一化二值，对称）
        self._latent_sem_adj_bin: Optional[np.ndarray] = None
        self._latent_sem_last_update: int = -1

        # [LAP] spatial Laplacian smoothness config
        self.gamma_smooth       = gamma_smooth
        self.smooth_start_epoch = smooth_start_epoch

        # [EXP] experiment / ablation controls
        valid_modes = {"easy_to_hard", "random", "hard_to_easy", "static_easy", "none"}
        if curriculum_mode not in valid_modes:
            raise ValueError(f"curriculum_mode must be one of {sorted(valid_modes)}, got {curriculum_mode!r}")
        self.curriculum_mode    = curriculum_mode
        self.record_diagnostics = bool(record_diagnostics)
        self.respect_input_adj  = bool(respect_input_adj)
        self.use_feature_perturbation = bool(use_feature_perturbation)
        self.history = {
            "epoch": [], "loss": [], "loss_recon": [], "loss_contrastive": [],
            "loss_edge": [], "edge_diff_mean": [], "masked_diff_mean": [],
            "n_masked": [], "lambda": [], "weight_v": [], "semantic_strength": [],
        }

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        # ---- 预处理（与原始流程相同）----
        from GraphST.preprocess import (
            preprocess, construct_interaction, add_contrastive_label, get_feature
        )
        if 'highly_variable' not in self.adata.var.keys():
            preprocess(self.adata)
        if 'adj' not in self.adata.obsm.keys():
            construct_interaction(self.adata, n_neighbors=n_spatial_neighbors)
        if 'label_CSL' not in self.adata.obsm.keys():
            add_contrastive_label(self.adata)
        if 'feat' not in self.adata.obsm.keys():
            get_feature(self.adata)

        self.features_np = self.adata.obsm['feat'].copy()
        self.features    = torch.FloatTensor(self.features_np).to(self.device)
        self.label_CSL   = torch.FloatTensor(
            self.adata.obsm['label_CSL']).to(self.device)

        n_spots          = self.adata.n_obs
        self.n_spots     = n_spots
        self.dim_input   = self.features.shape[1]

        # ----------------------------------------------------------------
        # [BUG-1 FIX] 保留**未归一化的二值** base_adj，用于后续在二值域上
        # 完成 "blend + edge mask"，最后才做对称归一化。
        # 原始代码先归一化再 mask，会破坏 GCN 的度归一化，是边课程在 ST 上
        # 失效的关键原因。
        # ----------------------------------------------------------------
        _base_adj_raw = self.adata.obsm['adj']
        if hasattr(_base_adj_raw, 'toarray'):
            _base_adj_dense = _base_adj_raw.toarray()
        else:
            _base_adj_dense = np.asarray(_base_adj_raw)
        self._base_adj_binary_np = _base_adj_dense.astype(np.float32).copy()

        self._base_adj = torch.FloatTensor(
            normalize_adj(self._base_adj_binary_np)
        ).to(self.device)
        self.graph_neigh = torch.FloatTensor(
            self.adata.obsm['graph_neigh'].copy() + np.eye(n_spots)
        ).to(self.device)

        # ---- 原始 curriculum / graph 组件 ----
        self.curriculum    = SelfPacedCurriculum(max_iter=epochs, K_max=K_max)
        self.graph_builder = AdaptiveGraphBuilder(
            spatial_coords=self.adata.obsm['spatial'],
            n_spatial_neighbors=n_spatial_neighbors,
            n_image_neighbors=n_image_neighbors,
            device=self.device,
            top_fraction=self.top_fraction,
        )
        # [EXP] Controlled graph-corruption experiments need every graph branch
        # to see the same perturbed adjacency rather than silently rebuilding a
        # clean KNN graph from coordinates.
        if self.respect_input_adj:
            self.graph_builder.spatial_adj = self._base_adj_binary_np.copy()
            self.graph_builder._spatial_neighbors = self.graph_builder._adj_to_neighbor_list(
                self.graph_builder.spatial_adj
            )

        # [MOD-1] 构建 edge_index，初始化 EdgeDifficultyEstimator
        spatial_adj_np  = self.adata.obsm['adj']
        self._edge_index_np = adj_to_edge_index(spatial_adj_np)  # (2, E)
        n_edges = self._edge_index_np.shape[1]

        self.edge_difficulty_estimator = EdgeDifficultyEstimator(   # [MOD-1]
            edge_index=self._edge_index_np,
            ema_decay=ema_decay,
        )

        # 同时保留原始 SpotConfidenceEstimator（供 AdaptiveGraphBuilder 使用）
        self.confidence_estimator = SpotConfidenceEstimator(
            n_spots, ema_decay=ema_decay, use_pearson=True
        )

        # [MOD-3] CurriculumEdgeMasker
        self.edge_masker = CurriculumEdgeMasker(        # [MOD-3]
            n_edges=n_edges,
            mask_ratio=mask_ratio,
            split_ratio=split_ratio,
        )
        self._mask_ratio  = mask_ratio
        self._split_ratio = split_ratio

        # edge_index tensor（放在 device 上，供 CrossCorrelationDecoder 使用）
        self._edge_index_t = torch.tensor(
            self._edge_index_np, dtype=torch.long
        ).to(self.device)

        # 用于 curriculum edge masking 的负样本边（随机采样等量负样本）
        #self._neg_edge_index_t = self._sample_neg_edges(n_neg=n_edges)

        # 空间 KNN 索引，用于 consistency 更新
        _nbrs_obj = NearestNeighbors(
            n_neighbors=n_spatial_neighbors + 1, algorithm='auto'
        ).fit(self.adata.obsm['spatial'])
        _, _nbr_ind = _nbrs_obj.kneighbors(self.adata.obsm['spatial'])
        self._spatial_nbr_idx = _nbr_ind[:, 1:].astype(np.int64)

        # ---- [UNI-REMOVED] 图像特征初始化已移除：本模型为纯基因表达模式 ----
        if image_embeddings is not None or image_patches is not None:
            print("[AdaMixGraphST] [UNI-REMOVED] image inputs are ignored "
                  "(gene-expression-only mode).")

        # [MOD-2] cc_decoder 在 train() 里才能知道 dim_output，这里先存 hidden_dim
        self._cc_hidden = cc_hidden

        # [MOD-4] PrototypeContrastiveLoss（dim_output 同样在 train() 时确认，先存参数）
        # proto_loss_fn 在 train() 开头实例化后加入 optimizer

        self._hard_neg_ei = self._build_2hop_neg_pool()

    # ------------------------------------------------------------------

    def _build_2hop_neg_pool(self) -> torch.Tensor:
        import scipy.sparse as sp
        adj = self.adata.obsm['adj']
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)

        adj_bin = (adj > 0).astype(np.float32)
        adj2    = (adj_bin @ adj_bin)           
        hard = (adj2 > 0).toarray() & (adj_bin.toarray() == 0)
        np.fill_diagonal(hard, False)

        rows, cols = np.where(hard)
        print(f"[AdaMixGraphST] Hard neg pool: {len(rows)} 2-hop pairs")

        min_needed = self._edge_index_np.shape[1]
        if len(rows) < min_needed:
            extra_t = self._sample_neg_edges(min_needed - len(rows))
            extra   = extra_t.cpu().numpy()
            rows    = np.concatenate([rows, extra[0]])
            cols    = np.concatenate([cols, extra[1]])

        neg_ei = np.stack([rows, cols], axis=0).astype(np.int64)
        return torch.tensor(neg_ei, dtype=torch.long).to(self.device)


    def _sample_neg_edges(self, n_neg: int) -> torch.Tensor:
        """
        随机采样 n_neg 条不存在边作为负样本（用于边 BCE 损失）。
        """
        adj_set = set(
            zip(self._edge_index_np[0].tolist(),
                self._edge_index_np[1].tolist())
        )
        rng     = np.random.default_rng(42)
        neg_src, neg_dst = [], []
        attempts = 0
        while len(neg_src) < n_neg and attempts < n_neg * 10:
            u = rng.integers(0, self.n_spots)
            v = rng.integers(0, self.n_spots)
            if u != v and (u, v) not in adj_set:
                neg_src.append(u)
                neg_dst.append(v)
            attempts += 1
        # 补充到 n_neg（如果负样本不足，重复最后一个）
        while len(neg_src) < n_neg:
            neg_src.append(neg_src[-1])
            neg_dst.append(neg_dst[-1])
        neg_ei = np.stack([neg_src[:n_neg], neg_dst[:n_neg]], axis=0)
        return torch.tensor(neg_ei, dtype=torch.long).to(self.device)

    # [UNI-REMOVED] _setup_image_features_precomputed() & _setup_image_features() deleted

    # ------------------------------------------------------------------
    def _get_adaptive_adj(self, confidence, mask_m, n_extra):
        #adj_np   = self.graph_builder.build_adj(confidence, mask_m, n_extra)
        adj_np = self.graph_builder.build_adj(confidence, mask_m, n_extra, max_img_dist=self.max_img_dist)
        adj_norm = normalize_adj(adj_np)
        return torch.FloatTensor(adj_norm).to(self.device)

    # ------------------------------------------------------------------
    def _blend_graphs(self, adaptive_adj, graph_strength):
        graph_strength = float(np.clip(graph_strength, 0.0, 1.0))
        return (1.0 - graph_strength) * self._base_adj + graph_strength * adaptive_adj

    # ------------------------------------------------------------------
    # [BUG-1 FIX] 新增：二值域邻接矩阵构建路径
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # [LAP] Spatial Laplacian smoothness regularization
    # ------------------------------------------------------------------
    @staticmethod
    def _laplacian_smoothness_loss(
        hiden_feat: torch.Tensor,
        spatial_nbr_idx: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        空间拉普拉斯平滑正则：让**空间相邻** spot 的 latent 表示接近。

        L_smooth = (1/(N*k)) * Σ_i Σ_{j∈KNN(i)} || z_i - z_j ||^2

        与 latent_sem / proto / DEC 等"离散化"损失的关键区别：
          - 本损失**只拉近空间局部邻居，不推开任何点**（无负样本项）
          - 因此它保持流形(manifold)的局部连续性，而不会把簇撕裂成离散块
          - 跨层相邻 spot 会被轻微拉近，但 reconstruction loss 会把它们
            按表达差异重新撑开 —— 两者平衡产生平滑的 trajectory 嵌入

        为避免 collapse（所有点挤成一团），本损失须与 reconstruction loss
        协同：reconstruction 负责"撑开"整体结构，smoothness 负责"局部平滑"。
        权重 gamma_smooth 不宜过大（建议 0.05~0.3）。

        Parameters
        ----------
        hiden_feat      : (N, d) GCN encoder 输出的 latent
        spatial_nbr_idx : (N, k) 每个 spot 的空间近邻索引（不含自身）
        normalize       : 是否对 latent 做 L2 归一化后再算距离。
                          归一化能让损失尺度与 latent 范数解耦、更稳定。

        Returns
        -------
        scalar loss
        """
        z = hiden_feat
        if normalize:
            z = F.normalize(z, dim=1)

        # gather 空间近邻的 latent：(N, k, d)
        z_nbr = z[spatial_nbr_idx]                       # 高级索引
        # 中心 spot 广播到 (N, 1, d)
        diff = z.unsqueeze(1) - z_nbr                    # (N, k, d)
        # 每条边的平方欧氏距离，对邻居和 spot 求均值
        loss = diff.pow(2).sum(dim=-1).mean()
        return loss

    # ------------------------------------------------------------------
    # [AHSG] Latent semantic neighbors (motivated by SpaCross AHSG)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_latent_semantic_adj(
        self,
        hiden_feat: torch.Tensor,
        epoch: int,
    ) -> None:
        """
        基于当前 latent (hiden_feat) 重新计算每个 spot 的语义近邻，
        并把它存为对称二值图，缓存到 self._latent_sem_adj_bin。

        与 SpaCross AHSG 的对应：
          - SpaCross 用 PCA / encoder 输出 + cosine top-K_intra 取语义邻居
          - 我们直接用 GCN encoder 的 latent，逻辑等价
          - 缓存 + 周期刷新，分摊计算成本（SpaCross 每 50 step 刷一次）

        被调用条件由 train 主循环决定（warmup 后 + refresh 周期到）。
        """
        z = F.normalize(hiden_feat, dim=1).detach().cpu().numpy()
        n = z.shape[0]
        k = min(self.latent_sem_k, n - 1)

        # cosine KNN（向量化）
        # 因为已 L2 normalize，cosine 相似 = 内积
        sim = z @ z.T
        np.fill_diagonal(sim, -np.inf)                          # 排除自己

        # 取 top-k：argpartition 比完整 argsort 快很多
        top_idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]   # (N, k)

        # 构造对称二值邻接（不带自环；GCN 归一化时统一加）
        adj = np.zeros((n, n), dtype=np.float32)
        rows = np.repeat(np.arange(n), k)
        cols = top_idx.reshape(-1)
        adj[rows, cols] = 1.0
        adj = np.clip(adj + adj.T, 0.0, 1.0)                    # 对称化

        self._latent_sem_adj_bin    = adj
        self._latent_sem_last_update = epoch

    # ------------------------------------------------------------------
    def _should_refresh_latent_sem(self, epoch: int) -> bool:
        """是否应该在当前 epoch 重算 latent 语义图。"""
        if not self.use_latent_semantic:
            return False
        if epoch < self.latent_sem_start_epoch:
            return False
        # 首次触发：start_epoch 时强制更新；之后按 refresh 周期
        if self._latent_sem_last_update < 0:
            return True
        return (epoch - self._latent_sem_last_update) >= self.latent_sem_refresh

    # ------------------------------------------------------------------
    def _get_adaptive_adj_binary(self, confidence, mask_m, n_extra) -> np.ndarray:
        """
        返回**未归一化的二值**自适应邻接矩阵 (numpy float32, dense)。

        与原 _get_adaptive_adj 的区别：不再调用 normalize_adj，留给
        _build_train_adj 在 mask 完成后统一归一化。
        """
        return self.graph_builder.build_adj(
            confidence, mask_m, n_extra, max_img_dist=self.max_img_dist
        ).astype(np.float32)

    # ------------------------------------------------------------------
    def _build_train_adj(
        self,
        adaptive_adj_bin: np.ndarray,
        graph_strength: float,
        masked_edge_idx: Optional[np.ndarray] = None,
        latent_sem_adj_bin: Optional[np.ndarray] = None,
        latent_sem_strength: float = 0.0,
    ) -> torch.Tensor:
        """
        [BUG-1 FIX] 在**未归一化二值**矩阵上完成 "融合 + 边掩码"，
        然后**最后**做一次对称归一化。

        原始流程：normalize_adj -> blend -> apply_mask（破坏度归一化）
        修复流程：blend -> apply_mask -> normalize_adj

        [AHSG] 现支持三路融合：
          adj = (1-g-s) * spatial_base + g * adaptive_image + s * latent_semantic
        其中
          g = graph_strength      （UNI image 边的占比，原有逻辑）
          s = latent_sem_strength （latent 语义邻居占比，新增）
        语义图 None 或权重为 0 时退化为原来的两路融合。

        Parameters
        ----------
        adaptive_adj_bin     : (N, N) 未归一化的 image-adaptive 二值矩阵
        graph_strength       : image-adaptive 图占比 ∈ [0, 1]
        masked_edge_idx      : 被选中遮蔽的边局部索引；None/空 表示不 mask
        latent_sem_adj_bin   : (N, N) latent 语义邻居二值图；None 表示不用
        latent_sem_strength  : latent 语义图占比 ∈ [0, 1]

        Returns
        -------
        adj : (N, N) 归一化后的 torch.FloatTensor
        """
        gs = float(np.clip(graph_strength, 0.0, 1.0))
        ss = float(np.clip(latent_sem_strength, 0.0, 1.0))

        # 防止两个外部权重之和超过 1（保留 spatial 至少 0 占比）
        total_external = gs + ss
        if total_external > 1.0:
            gs = gs / total_external
            ss = ss / total_external
            total_external = 1.0

        base_w = 1.0 - total_external

        # 1. 三路融合（仍在未归一化的二值域上）
        adj_blend = base_w * self._base_adj_binary_np + gs * adaptive_adj_bin
        if latent_sem_adj_bin is not None and ss > 0.0:
            adj_blend = adj_blend + ss * latent_sem_adj_bin

        # 2. 在未归一化矩阵上 mask 选中的边
        if masked_edge_idx is not None and len(masked_edge_idx) > 0:
            src = self._edge_index_np[0][masked_edge_idx]
            dst = self._edge_index_np[1][masked_edge_idx]
            adj_blend[src, dst] = 0.0
            adj_blend[dst, src] = 0.0

        # 3. 最后做对称归一化 D^{-1/2} (A+I) D^{-1/2}
        adj_norm = normalize_adj(adj_blend)
        return torch.FloatTensor(adj_norm).to(self.device)

    # ------------------------------------------------------------------
    def _apply_edge_mask(
        self,
        adj: torch.Tensor,
        masked_edge_idx: np.ndarray,
    ) -> torch.Tensor:
        """
        [MOD-3] 将选中的边从邻接矩阵中遮蔽（置零）。

        Parameters
        ----------
        adj             : (N, N) 归一化邻接矩阵
        masked_edge_idx : (K,)  被选中遮蔽的边局部索引（edge_index 的列号）
        """
        adj_masked = adj.clone()
        src = self._edge_index_np[0][masked_edge_idx]
        dst = self._edge_index_np[1][masked_edge_idx]
        adj_masked[src, dst] = 0.0
        adj_masked[dst, src] = 0.0
        return adj_masked

    # ------------------------------------------------------------------
    # [EXP] Curriculum strategy switch for controlled ablations
    def _select_experiment_mask_edges(self, difficulty, rng, epoch):
        mode = self.curriculum_mode
        if mode == "none":
            return np.empty(0, dtype=np.int64)

        # Keep the same number of masked edges across curriculum variants.
        self.edge_masker.update_lambda(epoch, self.epochs)
        K_target = int(np.ceil(self.edge_masker._lambda * self._mask_ratio * len(difficulty)))
        K_target = min(K_target, int(self._mask_ratio * len(difficulty)))
        K_target = max(K_target, 1)

        if mode == "random":
            return rng.choice(len(difficulty), size=K_target, replace=False).astype(np.int64)
        if mode == "static_easy":
            K_static = max(1, int(self._mask_ratio * len(difficulty)))
            return np.argsort(difficulty)[:K_static].astype(np.int64)
        if mode == "hard_to_easy":
            # Mirror of the proposed schedule: progressively include edges from
            # hardest toward easier ones. Same K_target, opposite ordering.
            return np.argsort(-difficulty)[:K_target].astype(np.int64)

        # proposed easy_to_hard (keeps split_ratio-guided + random mixture)
        return self.edge_masker.select_mask_edges(difficulty=difficulty, rng=rng)

    # ------------------------------------------------------------------
    def _compute_edge_recon_loss(self, cc_decoder, h_list, masked_edge_idx):
        if masked_edge_idx is None or len(masked_edge_idx) == 0:
            zero = torch.tensor(0.0, device=self.device)
            empty_logits = torch.empty(0, device=self.device)
            return zero, empty_logits, np.empty(0, dtype=np.int64)

        pos_src = torch.tensor(
            self._edge_index_np[0][masked_edge_idx], dtype=torch.long
        ).to(self.device)
        pos_dst = torch.tensor(
            self._edge_index_np[1][masked_edge_idx], dtype=torch.long
        ).to(self.device)
        pos_ei  = torch.stack([pos_src, pos_dst], dim=0)
        n_pos   = pos_ei.shape[1]

        pool_size = self._hard_neg_ei.shape[1]
        perm      = torch.randperm(pool_size, device=self.device)[:n_pos]
        neg_ei    = self._hard_neg_ei[:, perm]

        all_ei     = torch.cat([pos_ei, neg_ei], dim=1)
        all_labels = torch.cat([
            torch.ones(n_pos,  device=self.device),
            torch.zeros(n_pos, device=self.device),
        ])

        logits     = cc_decoder(h_list, all_ei)
        loss       = F.binary_cross_entropy_with_logits(logits, all_labels)
        pos_logits = logits[:n_pos].detach()

        return loss, pos_logits, masked_edge_idx

    # ------------------------------------------------------------------
    def train(self) -> object:
        """
        主训练循环。

        相比原始版本的变化（逐步标注）：
          [MOD-1] 用 edge_difficulty_estimator 替代 confidence_estimator 驱动课程
          [MOD-2] 新增 cc_decoder，计算边重建 BCE 损失
          [MOD-3] 新增 edge_masker，对训练图施加"先易后难"边遮蔽
        """
        from GraphST.model import Encoder
        from tqdm import tqdm

        model = Encoder(
            self.dim_input, self.dim_output, self.graph_neigh
        ).to(self.device)
        loss_CSL = nn.BCEWithLogitsLoss()
        self.topo_reg_fn = GraphTopologyRegularization(align_weight=1.0, sparse_weight=0.01).to(self.device)

        # [MOD-2] 初始化 CrossCorrelationDecoder（此时才知道 dim_output）
        cc_decoder = CrossCorrelationDecoder(          # [MOD-2]
            embed_dim=self.dim_output,
            n_layers=1,
            hidden_dim=self._cc_hidden,
        ).to(self.device)

        # [MOD-4] 初始化 PrototypeContrastiveLoss
        proto_loss_fn = PrototypeContrastiveLoss(      # [MOD-4]
            n_clusters=self._n_clusters,
            embed_dim=self.dim_proto,
            temperature=self._proto_temperature,
            uniform_weight=self._proto_uniform_w,
            spatial_weight=self._proto_spatial_w,
        ).to(self.device)

        # 联合优化 GNN encoder + edge decoder + 原型
        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(cc_decoder.parameters())
            + list(proto_loss_fn.parameters()),        # [MOD-4] 原型也参与优化
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        print("== AdaMixGraphST (+ Cur-MGAE) training started ==")

        features_a    = self.features
        final_adj_used = self._base_adj
        rng = np.random.default_rng(42)

        # spatial_nbr_idx tensor（供 PrototypeContrastiveLoss 空间平滑正则使用）
        _spatial_nbr_t = torch.tensor(
            self._spatial_nbr_idx, dtype=torch.long
        ).to(self.device)

        for epoch in tqdm(range(self.epochs)):
            model.train()
            cc_decoder.train()
            proto_loss_fn.train()                                   # [MOD-4]

            # ================================================================
            # Step 1: Curriculum 信号
            #   [MOD-1] 用 edge 难度聚合为 spot 置信度（替代纯 MSE 置信度）
            # ================================================================
            lam = self.curriculum.age_param(epoch)

            # [MOD-1] spot 置信度由边难度聚合得来

            #spot_conf = self.edge_difficulty_estimator.get_spot_confidence(
            #    self.n_spots
            #)
            #spot_conf = self.confidence_estimator.get_confidence()
            old_conf = self.confidence_estimator.get_confidence()
            edge_conf = self.edge_difficulty_estimator.get_spot_confidence(self.n_spots)
            #spot_conf = 0.5 * old_conf + 0.5 * edge_conf
            spot_conf = edge_conf
            self.confidence_estimator.fused_confidence = spot_conf
            # 也更新旧版 confidence_estimator（供 AdaptiveGraphBuilder 使用）
            # —— 此处直接将 edge 聚合的置信度写入，不再单独计算 MSE

            
            #self.confidence_estimator.fused_confidence = spot_conf

            L_bar = self.curriculum.compute_proxy_loss(spot_conf)

            if epoch < self.warmup_epochs:
                mask_m          = 0
                weight_v        = 0.0
                n_extra         = 0
                perturb_fraction = 0.0
                graph_strength  = 0.0
            else:
                raw_weight_v = self.curriculum.get_weight(L_bar, lam)
                ramp_epochs  = 150
                progress_after_warmup = (
                    (epoch - self.warmup_epochs) / max(ramp_epochs, 1)
                )
                max_strength = float(np.clip(progress_after_warmup, 0.0, 1.0))
                min_strength = 0.2 * max_strength
                #min_strength = 0.10 * max_strength

                weight_v = min(raw_weight_v, max_strength)
                if max_strength > 0.0:
                    weight_v = max(weight_v, min_strength)

                #mask_m          = 1 if weight_v >= 0.18 else 0
                direction_soft = 0.1
                mask_m = 1 if weight_v >= direction_soft else 0

                n_extra         = self.curriculum.n_perturbations(weight_v)
                perturb_fraction = min(0.30, 0.30 * weight_v)
                graph_strength  = min(0.30, 0.30 * weight_v)

                #perturb_fraction = min(0.10, 0.10 * weight_v)
                #graph_strength  = min(0.10, 0.0 * weight_v)
                #mask_m          = 0
                #n_extra         = 0
                #perturb_fraction = 0
                #graph_strength  = 0

            # ================================================================
            # [MOD-3] Step 2: CurriculumEdgeMasker 调度边遮蔽
            #   更新 λ 并选择本轮要遮蔽的边
            # ================================================================
            self.edge_masker.update_lambda(epoch, self.epochs)          # [MOD-3]
            edge_difficulty = self.edge_difficulty_estimator.get_difficulty()
            masked_edge_idx = self._select_experiment_mask_edges(
                difficulty=edge_difficulty, rng=rng, epoch=epoch
            )

            # ================================================================
            # Step 3: 构建自适应邻接矩阵
            # [BUG-1 FIX] 改为 "二值域 blend + mask -> 最后统一归一化"
            # 原来的 normalize -> blend -> mask 会破坏 GCN 度归一化
            # [AHSG] 额外引入 latent 语义邻居作为第三路融合源
            # ================================================================
            adaptive_adj_bin = self._get_adaptive_adj_binary(
                spot_conf, mask_m, n_extra
            )

            # [AHSG] 决定本 epoch 是否启用 latent 语义图
            # 缓存为 None（warmup 前 / 首次刷新前）时退化为原两路融合
            cur_sem_strength = (
                self.latent_sem_strength
                if (self.use_latent_semantic
                    and epoch >= self.latent_sem_start_epoch
                    and self._latent_sem_adj_bin is not None)
                else 0.0
            )

            # 训练用图（已 mask 选中边、再归一化）
            adj_masked = self._build_train_adj(
                adaptive_adj_bin,
                graph_strength,
                masked_edge_idx=masked_edge_idx,
                latent_sem_adj_bin=self._latent_sem_adj_bin,
                latent_sem_strength=cur_sem_strength,
            )

            # 诊断用图（无 mask、再归一化），用于 confidence_estimator 更新
            adj_used = self._build_train_adj(
                adaptive_adj_bin,
                graph_strength,
                masked_edge_idx=None,
                latent_sem_adj_bin=self._latent_sem_adj_bin,
                latent_sem_strength=cur_sem_strength,
            )

            # ================================================================
            # Step 4: 特征扰动（原始逻辑）
            # ================================================================
            if self.use_feature_perturbation and perturb_fraction > 0.0:
                feat_a_np = self.graph_builder.build_perturbed_features(
                    features=self.features_np,
                    confidence=spot_conf,
                    mask_m=mask_m,
                    n_swaps=max(1, n_extra),
                    mix_ratio_easy=0.15 + 0.10 * weight_v,
                    mix_ratio_hard=0.30 + 0.20 * weight_v,
                    #mix_ratio_easy=0.10 + 0.10 * weight_v,
                    #mix_ratio_hard=0.80 + 0.80 * weight_v,
                    perturb_fraction=perturb_fraction,
                )
                features_a = torch.FloatTensor(feat_a_np).to(self.device)
            else:
                features_a = self.features

            # ================================================================
            # Step 5: 前向传播（用遮蔽后的邻接矩阵）
            # ================================================================
            hiden_feat, emb, ret, ret_a = model(
                self.features, features_a, adj_masked  # [MOD-3] adj_masked
            )

            # [AHSG] 周期性刷新 latent 语义邻居图
            # 在 forward 之后做，使用本 epoch 已更新的 latent；下一 epoch 才会
            # 在 _build_train_adj 中真正用到（自然形成 "本轮算、下轮用" 的解耦）。
            if self._should_refresh_latent_sem(epoch):
                self._update_latent_semantic_adj(hiden_feat, epoch)

            if (epoch == self.warmup_epochs and not proto_loss_fn.initialized):
                proto_loss_fn.initialize_prototypes(emb.detach())

            #if self.graph_builder.image_adj is not None and epoch >= self.warmup_epochs:
            #    adj_image_norm = torch.FloatTensor(
            #        normalize_adj(self.graph_builder.image_adj)
            #    ).to(self.device)
            #    
            #    z_image, _, _, _ = model(self.features, self.features, adj_image_norm)

            #    loss_topo = self.topo_reg_fn(adj_used, adj_image_norm)
                
            #    loss = loss + 0.5 * loss_topo

            # ================================================================
            # Step 6: 计算损失
            #   原始损失：reconstruction MSE + contrastive BCE
            #   [MOD-2] 新增：边重建 BCE（CrossCorrelationDecoder）
            # ================================================================
            loss_sl_1  = loss_CSL(ret,   self.label_CSL)
            loss_sl_2  = loss_CSL(ret_a, self.label_CSL)
            loss_recon = F.mse_loss(self.features, emb)

            # [MOD-2] 边重建损失（只对被遮蔽的边计算 BCE）
            h_list = [hiden_feat]                                # K=1 层
            loss_edge, pos_logits, masked_edge_idx = self._compute_edge_recon_loss(
                cc_decoder, h_list, masked_edge_idx
            )

            # [MOD-4] 聚类损失（warmup 阶段关闭，encoder 稳定后才激活）
            # 使用 hiden_feat（latent z）而非 emb（重构输出），直接约束 latent 空间
            gamma_cluster_eff = (
                self.gamma_cluster if epoch >= self.warmup_epochs else 0.0
            )
            if gamma_cluster_eff > 0.0:
                loss_cluster = proto_loss_fn(
                    #hiden_feat,
                    emb,
                    spatial_nbr_idx=_spatial_nbr_t,
                )
            else:
                loss_cluster = torch.tensor(0.0, device=self.device)

            # [LAP] 空间拉普拉斯平滑损失（保持 latent manifold 连续性）
            # 只拉近空间近邻、不推开异簇，对抗 latent_sem/proto 造成的离散化，
            # 让 UMAP 呈现连续 trajectory 而非孤立块。
            gamma_smooth_eff = (
                self.gamma_smooth if epoch >= self.smooth_start_epoch else 0.0
            )
            if gamma_smooth_eff > 0.0:
                loss_smooth = self._laplacian_smoothness_loss(
                    hiden_feat, _spatial_nbr_t, normalize=True
                )
            else:
                loss_smooth = torch.tensor(0.0, device=self.device)

            # 总损失 = 原始损失 + 边重建损失 + 聚类损失 + 平滑损失
            loss = (
                self.alpha  * loss_recon
                + self.beta * (loss_sl_1 + loss_sl_2)
                + self.gamma_edge    * loss_edge         # [MOD-2]
                + gamma_cluster_eff  * loss_cluster      # [MOD-4]
                + gamma_smooth_eff   * loss_smooth       # [LAP]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ================================================================
            # Step 7: 更新难度估计器
            #   [MOD-1] 用 pos_logits 更新 EdgeDifficultyEstimator
            #   原始 SpotConfidenceEstimator 的 MSE 更新也保留（用于 log）
            # ================================================================
            with torch.no_grad():
                model.eval()
                cc_decoder.eval()
                proto_loss_fn.eval()                                 # [MOD-4]

                # [MOD-1] 更新边难度（只更新本轮被遮蔽的边）
                if len(masked_edge_idx) > 0:
                    self.edge_difficulty_estimator.update(             # [MOD-1]
                        pos_logits, edge_indices=masked_edge_idx
                    )

                # 保留 MSE 置信度更新（仅用于 log 诊断）
                h_eval, emb_eval, _, _ = model(
                    self.features, features_a, adj_used
                )
                self.confidence_estimator.update(self.features, emb_eval)
                self.confidence_estimator.update_with_spatial_consistency(
                    h_eval.detach(),
                    self._spatial_nbr_idx,
                    consistency_weight=self.consistency_weight,
                )
                model.train()
                cc_decoder.train()
                proto_loss_fn.train()                                # [MOD-4]

            final_adj_used = adj_used

            # [EXP] Training dynamics for mechanism figures.
            if self.record_diagnostics:
                masked_diff_mean = (
                    float(edge_difficulty[masked_edge_idx].mean())
                    if len(masked_edge_idx) > 0 else np.nan
                )
                self.history["epoch"].append(int(epoch))
                self.history["loss"].append(float(loss.item()))
                self.history["loss_recon"].append(float(loss_recon.item()))
                self.history["loss_contrastive"].append(float((loss_sl_1 + loss_sl_2).item()))
                self.history["loss_edge"].append(float(loss_edge.item()))
                self.history["edge_diff_mean"].append(float(edge_difficulty.mean()))
                self.history["masked_diff_mean"].append(masked_diff_mean)
                self.history["n_masked"].append(int(len(masked_edge_idx)))
                self.history["lambda"].append(float(self.edge_masker._lambda))
                self.history["weight_v"].append(float(weight_v))
                self.history["semantic_strength"].append(float(cur_sem_strength))

            if epoch % 50 == 0:
                diff_mean = float(edge_difficulty.mean())
                n_masked  = len(masked_edge_idx)
                # [AHSG] latent semantic diagnostic
                if self._latent_sem_adj_bin is not None:
                    sem_n_edges = int(self._latent_sem_adj_bin.sum() / 2)
                    sem_info = (f", sem_strength={cur_sem_strength:.3f}, "
                                f"sem_n_edges={sem_n_edges}, "
                                f"sem_last_upd={self._latent_sem_last_update}")
                else:
                    sem_info = ", sem=disabled"
                print(
                    f"epoch {epoch}: loss={loss.item():.4f}, "
                    f"loss_recon={loss_recon.item():.4f}, "
                    f"loss_edge={loss_edge.item():.4f}, "
                    f"loss_cluster={loss_cluster.item():.4f}, "  # [MOD-4]
                    f"loss_smooth={loss_smooth.item():.4f}, "    # [LAP]
                    f"edge_diff_mean={diff_mean:.4f}, "
                    f"n_masked={n_masked}, "
                    f"mask_m={mask_m}, weight_v={weight_v:.4f}"
                    f"{sem_info}"
                )

        print("== Optimisation finished ==")

        with torch.no_grad():
            model.eval()
            emb_rec = model(
                self.features, features_a, final_adj_used
            )[1].detach().cpu().numpy()

        self.adata.obsm['emb'] = emb_rec
        # [EXP] expose diagnostics without changing the standard return API
        self.final_edge_difficulty_ = self.edge_difficulty_estimator.get_difficulty().copy()
        self.final_edge_index_ = self._edge_index_np.copy()
        if self.record_diagnostics:
            self.adata.uns['adamix_history'] = {
                k: np.asarray(v) for k, v in self.history.items()
            }
            self.adata.uns['adamix_experiment'] = {
                'curriculum_mode': self.curriculum_mode,
                'mask_ratio': float(self._mask_ratio),
                'split_ratio': float(self._split_ratio),
                'use_latent_semantic': bool(self.use_latent_semantic),
                'latent_sem_strength': float(self.latent_sem_strength),
                'gamma_edge': float(self.gamma_edge),
                'beta': float(self.beta),
                'use_feature_perturbation': bool(self.use_feature_perturbation),
            }
        return self.adata

def extract_visium_patches(adata, patch_size=64, image_key='hires'):
    try:
        library_id = list(adata.uns['spatial'].keys())[0]
        img_dict   = adata.uns['spatial'][library_id]['images']
        scale_dict = adata.uns['spatial'][library_id]['scalefactors']
        if image_key not in img_dict:
            warnings.warn(f"Image key '{image_key}' not found.")
            return None
        img = img_dict[image_key]
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        scale      = scale_dict.get(f'tissue_{image_key}_scalef', 1.0)
        coords_px  = (adata.obsm['spatial'] * scale).astype(int)
        half       = patch_size // 2
        H, W       = img.shape[:2]
        n_spots    = len(coords_px)
        patches    = np.zeros((n_spots, 3, patch_size, patch_size), dtype=np.uint8)
        for idx, (cx, cy) in enumerate(coords_px):
            r0 = np.clip(cy - half, 0, H); r1 = np.clip(cy + half, 0, H)
            c0 = np.clip(cx - half, 0, W); c1 = np.clip(cx + half, 0, W)
            crop = img[r0:r1, c0:c1, :]
            pad_h = patch_size - crop.shape[0]
            pad_w = patch_size - crop.shape[1]
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)),
                              mode='constant', constant_values=128)
            patches[idx] = crop[:patch_size, :patch_size, :].transpose(2, 0, 1)
        print(f"[extract_visium_patches] Extracted {n_spots} patches "
              f"of size {patch_size}×{patch_size}.")
        return patches
    except (KeyError, IndexError, TypeError) as e:
        warnings.warn(f"[extract_visium_patches] Could not extract: {e}")
        return None


# =============================================================================
# Smoke-test
# =============================================================================

if __name__ == '__main__':
    print("=== Smoke-test: AdaMixGraphST (+Cur-MGAE) with synthetic data ===")
    try:
        import anndata as ad
        import scipy.sparse as sp

        n_spots, n_genes = 200, 300
        X     = np.random.rand(n_spots, n_genes).astype(np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(X))
        adata.obsm['spatial'] = np.random.rand(n_spots, 2) * 1000.0
        adata.var['highly_variable'] = True

        fake_patches = np.random.randint(
            0, 255, (n_spots, 3, 64, 64), dtype=np.uint8
        )

        device = torch.device('cpu')
        model  = AdaMixGraphST(
            adata,
            image_patches=fake_patches,
            epochs=10,
            device=device,
            dim_output=32,
            K_max=3,
            use_pretrained_cnn=False,
            random_seed=0,
            mask_ratio=0.5,
            split_ratio=0.5,
            gamma_edge=1.0,
            n_clusters=5,
            gamma_cluster=0.5,
            proto_temperature=0.2,
        )
        result_adata = model.train()
        emb = result_adata.obsm['emb']
        assert emb.shape == (n_spots, 32), f"Unexpected shape: {emb.shape}"
        print(f"OK — embedding shape: {emb.shape}")
    except ImportError as e:
        print(f"Skipping smoke-test (missing dependency: {e})")