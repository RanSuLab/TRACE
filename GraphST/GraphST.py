import torch
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
from .model import Encoder, Encoder_sparse, Encoder_map, Encoder_sc
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd


# =============================================================================
# [REFINE] Identity-initialized Spatial Attention Refinement Head
# =============================================================================
class RefinementHead(nn.Module):
    """
    轻量、identity 初始化的 latent 精炼头，用空间注意力温和地重组每个 spot 的表示。

    设计目标：对抗"语义偏移/断层"，但**不引入漂移**。三道保险：
      1. identity 初始化：输出 = z + alpha * Δ，alpha 初值=0 → 初期完全不改变 z
      2. 残差 + 全局门控 alpha（可学习标量）：精炼是温和的残差修正，强度从 0 慢慢学
      3. 参数量极小：只有 Q/K/V 三个小投影 + 一个标量门

    注意力为什么对症断层：
      固定图传播 mm(adj,z) 对所有邻居一视同仁 → 跨层邻居污染。
      可学习注意力能压低跨层邻居权重、提高同层邻居权重，自适应地把裂开的
      同层 spot 通过邻居重新聚合，而不模糊跨层边界。

    Parameters
    ----------
    dim       : latent 维度
    attn_dim  : 注意力内部投影维度（小，控制参数量）
    """
    def __init__(self, dim: int, attn_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.attn_dim = attn_dim
        self.q = nn.Linear(dim, attn_dim, bias=False)
        self.k = nn.Linear(dim, attn_dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        # [identity init] 全局门控标量，初值 0 → 初期 refine 输出 == 输入 z
        self.alpha = nn.Parameter(torch.zeros(1))
        # 注意：v 用正常初始化（不能也置 0，否则 delta 恒为 0、alpha 拿不到梯度）。
        # identity 行为仅由 alpha=0 保证；alpha 一旦获得梯度即可学习精炼强度。
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, z, nbr_idx, nbr_mask):
        """
        z        : (N, d) latent
        nbr_idx  : (N, K) 每个 spot 的空间邻居索引（不足处用 0 填充）
        nbr_mask : (N, K) bool，True=有效邻居（区分填充位）
        return   : (N, d) 精炼后的 latent
        """
        N, K = nbr_idx.shape
        q = self.q(z)                                   # (N, attn_dim)
        k_all = self.k(z)                               # (N, attn_dim)
        v_all = self.v(z)                               # (N, d)

        k_nbr = k_all[nbr_idx]                          # (N, K, attn_dim)
        v_nbr = v_all[nbr_idx]                          # (N, K, d)

        # 注意力打分：q 和每个邻居 k 的点积
        scores = (q.unsqueeze(1) * k_nbr).sum(-1) / (self.attn_dim ** 0.5)  # (N, K)
        scores = scores.masked_fill(~nbr_mask, float('-inf'))
        attn = torch.softmax(scores, dim=1)             # (N, K)
        # 防止全 -inf 行（无邻居）产生 nan
        attn = torch.nan_to_num(attn, nan=0.0)

        delta = (attn.unsqueeze(-1) * v_nbr).sum(1)     # (N, d)
        # 残差 + 全局门控（identity init: alpha=0 → 返回 z）
        return z + self.alpha * delta


    
class GraphST():
    def __init__(self, 
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        random_seed = 41,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X',
        # ============ [NEG] debiased node-to-node 对比 (方案1) ============
        gamma_neg = 0.0,            # 新对比损失权重；0=完全关闭，退化为原版 GraphST
        neg_temperature = 0.5,      # InfoNCE 温度
        neg_n_neg = 256,            # 每个 spot 的随机负样本数
        neg_debias_spatial = True,  # 排除空间邻居作负样本（假负样本）
        neg_debias_pseudo = False,  # 用 latent KMeans 伪标签排除同簇负样本
        neg_pseudo_k = 7,           # 伪标签聚类数
        neg_pseudo_refresh = 50,    # 每多少 epoch 刷新一次伪标签
        neg_start_epoch = 0,        # 从第几个 epoch 开始加该损失
        # ===== [NEG-S3] 语义正样本（专治断层：把裂开的同层两块拉回） =====
        neg_semantic_pos = False,   # 正样本除空间邻居外，加入 latent 高相似 spot
        neg_sem_pos_k = 10,         # 每个 spot 取 top-k 个 latent 相似 spot 作正样本
        neg_sem_pos_start = 200,    # 从第几 epoch 开始用语义正样本（需 latent 先稳定）
        # ===== [PLR] Pseudo-label Refinement（无监督，抗语义偏移）=====
        gamma_plr = 0.0,            # 伪标签精炼损失权重；0=关闭
        plr_k = 7,                  # 伪标签聚类数（=n_clusters）
        plr_start_epoch = 300,      # 从第几 epoch 开始（latent 需先稳定）
        plr_refresh = 20,           # 每多少 epoch 刷新一次伪标签
        plr_conf_quantile = 0.5,    # 置信度过滤：只用置信度高于该分位数的 spot
        plr_spatial_agree = 0.5,    # 空间一致性过滤：伪标签须与≥该比例空间邻居一致
        plr_temp = 1.0,             # 伪标签 soft-assignment 温度
        # ===== [REFINE] Identity-init spatial attention refinement head =====
        use_refine = False,         # 是否启用精炼头
        refine_attn_dim = 32,       # 注意力内部维度
        refine_n_nbr = 9,           # 每个 spot 用多少空间邻居做注意力
        refine_lr = None,           # 精炼头单独学习率（None=用主 lr）
        ):
        '''\
        (docstring 同原版，省略)
        '''
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype

        # [NEG] 保存方案1配置
        self.gamma_neg          = gamma_neg
        self.neg_temperature    = neg_temperature
        self.neg_n_neg          = neg_n_neg
        self.neg_debias_spatial = neg_debias_spatial
        self.neg_debias_pseudo  = neg_debias_pseudo
        self.neg_pseudo_k       = neg_pseudo_k
        self.neg_pseudo_refresh = neg_pseudo_refresh
        self.neg_start_epoch    = neg_start_epoch
        # [NEG-S3] 语义正样本配置
        self.neg_semantic_pos   = neg_semantic_pos
        self.neg_sem_pos_k      = neg_sem_pos_k
        self.neg_sem_pos_start  = neg_sem_pos_start
        self._neg_epoch         = 0          # 当前 epoch（供损失函数判断阶段）
        self._neg_pseudo_labels = None      # 缓存伪标签
        # [PLR] pseudo-label refinement config
        self.gamma_plr          = gamma_plr
        self.plr_k              = plr_k
        self.plr_start_epoch    = plr_start_epoch
        self.plr_refresh        = plr_refresh
        self.plr_conf_quantile  = plr_conf_quantile
        self.plr_spatial_agree  = plr_spatial_agree
        self.plr_temp           = plr_temp
        self._plr_centroids     = None       # 簇中心（torch）
        self._plr_labels        = None       # 精炼后的伪标签
        self._plr_trusted       = None       # 可信 spot mask（bool）
        self._plr_prev_labels   = None       # 上一次伪标签（时间稳定性过滤）
        # [REFINE] refinement head config
        self.use_refine         = use_refine
        self.refine_attn_dim    = refine_attn_dim
        self.refine_n_nbr       = refine_n_nbr
        self.refine_lr          = refine_lr
        self._refine_nbr_idx    = None       # (N, K) 空间邻居索引（train 时构建）
        self._refine_nbr_mask   = None       # (N, K) 有效邻居 mask
        self.refine_head        = None       # RefinementHead 实例（train 时构建）
        
        fix_seed(self.random_seed)
        
        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata)
        
        if 'adj' not in adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide']:
              construct_interaction_KNN(self.adata)
           else:    
              construct_interaction(self.adata)
         
        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
    
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        if self.datatype in ['Stereo', 'Slide']:
           #using sparse
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           # standard version
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
        
        if self.deconvolution:
           self.adata_sc = adata_sc.copy() 
            
           if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
              self.feat_sp = adata.X.toarray()[:, ]
           else:
              self.feat_sp = adata.X[:, ]
           if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
              self.feat_sc = self.adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = self.adata_sc.X[:, ]
            
           # fill nan as 0
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
          
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
        
           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 

           self.n_cell = adata_sc.n_obs
        self.n_spot = adata.n_obs

    # ====================================================================
    # [NEG] 方案1：debiased node-to-node 对比损失
    # ====================================================================
    def _refresh_pseudo_labels(self, hiden_feat):
        """用当前 latent 做一次 KMeans，得到伪标签（供 debias 排除同簇负样本）。"""
        from sklearn.cluster import KMeans
        z = F.normalize(hiden_feat, dim=1).detach().cpu().numpy()
        km = KMeans(n_clusters=self.neg_pseudo_k, n_init=10,
                    random_state=self.random_seed).fit(z)
        self._neg_pseudo_labels = torch.as_tensor(
            km.labels_, dtype=torch.long, device=self.device
        )

    def _refined_contrastive_loss(self, hiden_feat, neg_rng):
        """
        node-to-node InfoNCE，精细负样本选择，解决 GraphST DGI 负样本粗糙问题：
          正样本: 空间邻居 (graph_neigh)，大概率同层
                  [NEG-S3] (可选) + latent 高相似 spot，把裂开的同层两块拉回（治断层）
          负样本: 随机采样，排除 (a) 空间邻居 (b) 同伪标签 的假负样本(debias)
        """
        z = F.normalize(hiden_feat, dim=1)
        N = z.size(0)
        idx = torch.arange(N, device=self.device)
        sim_all = z @ z.T / self.neg_temperature             # (N, N)

        # ---- 正样本 mask：空间邻居（去自环）----
        gn = self.graph_neigh
        pos_mask = (gn > 0).clone()
        pos_mask[idx, idx] = False

        # ---- [NEG-S3] 语义正样本：latent 高相似 spot ----
        # 动机：断层 = 同一层带在 latent 里裂成两块。空间 1-hop 邻居只能局部
        #       平滑，连不上裂缝两侧的大块。加入"latent 高相似 spot"作正样本，
        #       把本质同层、但空间不相邻的两块重新拉到一起。
        #       关键：仅在 latent 稳定后启用，且这些相似 spot 会经 debias 不被
        #       当负样本，从而软拉近（区别于 KMeans 硬切分制造断层）。
        if (self.neg_semantic_pos and
                self._neg_epoch >= self.neg_sem_pos_start):
            k_sem = min(self.neg_sem_pos_k, N - 1)
            # 用未缩放的 cosine 取 top-k（sim_all 已含温度，topk 顺序不变）
            with torch.no_grad():
                sem_sim = sim_all.clone()
                sem_sim[idx, idx] = float('-inf')
                sem_topk = sem_sim.topk(k_sem, dim=1).indices       # (N, k_sem)
            sem_pos = torch.zeros_like(pos_mask)
            sem_pos[idx.unsqueeze(1), sem_topk] = True
            pos_mask = pos_mask | sem_pos

        has_pos = pos_mask.sum(1) > 0

        # ---- 假负样本 mask：空间邻居 + 自身 (+ 同伪标签) (+ 语义正样本) ----
        exclude = torch.zeros((N, N), dtype=torch.bool, device=self.device)
        if self.neg_debias_spatial:
            exclude = exclude | (gn > 0)
        exclude[idx, idx] = True
        if self.neg_debias_pseudo and self._neg_pseudo_labels is not None:
            pl = self._neg_pseudo_labels
            exclude = exclude | (pl.unsqueeze(0) == pl.unsqueeze(1))
        # [NEG-S3] 语义正样本也不能当负样本（否则自相矛盾：既拉近又推开）
        exclude = exclude | pos_mask

        # ---- 负样本：随机采 n_neg，过滤假负 ----
        n_neg = min(self.neg_n_neg, N - 1)
        neg_idx = torch.randint(0, N, (N, n_neg), device=self.device, generator=neg_rng)
        is_fake = exclude[idx.unsqueeze(1), neg_idx]          # (N, n_neg)
        neg_sim = torch.gather(sim_all, 1, neg_idx)           # (N, n_neg)
        neg_sim = neg_sim.masked_fill(is_fake, float('-inf'))
        neg_lse = torch.logsumexp(neg_sim, dim=1, keepdim=True)   # (N,1)

        # ---- 正样本聚合 logsumexp ----
        pos_sim = sim_all.masked_fill(~pos_mask, float('-inf'))
        pos_lse = torch.logsumexp(pos_sim, dim=1, keepdim=True)   # (N,1)

        denom = torch.logaddexp(pos_lse, neg_lse)
        per_node = -(pos_lse - denom).squeeze(1)
        valid = has_pos & torch.isfinite(per_node)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return per_node[valid].mean()

    # ====================================================================
    # [PLR] Pseudo-label Refinement（无监督，抗语义偏移）
    # ====================================================================
    @torch.no_grad()
    def _refresh_plr(self, hiden_feat):
        """
        刷新精炼伪标签。核心：不盲信伪标签，用三重过滤只保留"可信" spot。

        与之前 KMeans 约束（放大断层）的本质区别：
          KMeans 约束无差别相信所有伪标签 → 强化模糊区/断层处的错误。
          PLR 三重过滤后只约束"高置信 + 空间一致 + 时间稳定"的 spot，
          模糊区/断层处的 spot 被过滤掉、不参与约束 → 不放大错误。
        """
        from sklearn.cluster import KMeans
        z = F.normalize(hiden_feat, dim=1)
        z_np = z.detach().cpu().numpy()

        # 1) KMeans 得到伪标签 + 簇中心
        km = KMeans(n_clusters=self.plr_k, n_init=10,
                    random_state=self.random_seed).fit(z_np)
        labels = torch.as_tensor(km.labels_, dtype=torch.long, device=self.device)
        centroids = F.normalize(
            torch.as_tensor(km.cluster_centers_, dtype=torch.float32,
                            device=self.device), dim=1
        )                                                    # (K, d)

        # 2) 置信度：spot 到其所属簇中心的 cosine 相似度
        sim_to_centroid = (z * centroids[labels]).sum(1)     # (N,)
        conf_thresh = torch.quantile(sim_to_centroid, self.plr_conf_quantile)
        conf_ok = sim_to_centroid >= conf_thresh

        # 3) 空间一致性：伪标签与空间邻居多数一致（向量化）
        gn = (self.graph_neigh > 0).float()
        idx = torch.arange(z.size(0), device=self.device)
        gn = gn.clone(); gn[idx, idx] = 0.0
        # one-hot 伪标签 → 邻居中各类计数 → 与自身标签一致的比例
        onehot = F.one_hot(labels, num_classes=self.plr_k).float()   # (N, K)
        nbr_label_count = gn @ onehot                                # (N, K)
        nbr_total = gn.sum(1).clamp_min(1.0)                          # (N,)
        agree = nbr_label_count[idx, labels] / nbr_total             # (N,)
        spatial_ok = (gn.sum(1) > 0) & (agree >= self.plr_spatial_agree)

        # 4) 时间稳定性过滤：默认禁用。
        #    KMeans 每次 init 的簇 id 会任意置换（cluster 0 这次可能是上次的 3），
        #    直接比 id 会大面积误判"不稳定"，反而误伤可信 spot。
        #    置信度 + 空间一致性两个过滤已足够稳健，故不启用 id 比较。
        stable_ok = torch.ones_like(conf_ok)

        trusted = conf_ok & spatial_ok & stable_ok

        self._plr_centroids   = centroids
        self._plr_labels      = labels
        self._plr_trusted     = trusted
        self._plr_prev_labels = labels.clone()

        n_trust = int(trusted.sum().item())
        print(f"[PLR] refreshed pseudo-labels: "
              f"trusted {n_trust}/{z.size(0)} "
              f"({100.0*n_trust/z.size(0):.1f}%)  "
              f"[conf>{conf_thresh:.3f}, spatial_agree>={self.plr_spatial_agree}]")

    def _plr_loss(self, hiden_feat):
        """
        伪标签精炼损失：只对"可信" spot，拉近它和其所属簇中心。
        soft cross-entropy 形式，温度 plr_temp。仅 trusted spot 贡献梯度。
        """
        if self._plr_centroids is None or self._plr_trusted is None:
            return torch.tensor(0.0, device=self.device)
        z = F.normalize(hiden_feat, dim=1)
        logits = (z @ self._plr_centroids.T) / self.plr_temp   # (N, K)
        ce = F.cross_entropy(logits, self._plr_labels, reduction='none')  # (N,)
        m = self._plr_trusted.float()
        if m.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return (ce * m).sum() / m.sum()

    def _build_refine_neighbors(self):
        """
        [REFINE] 从 graph_neigh（空间邻接）构建每个 spot 的定长空间邻居索引，
        供 RefinementHead 的注意力使用。不足 refine_n_nbr 的用 0 填充并 mask。
        """
        gn = (self.graph_neigh > 0).detach().cpu().numpy()
        np.fill_diagonal(gn, False)
        N = gn.shape[0]
        K = self.refine_n_nbr
        nbr_idx  = np.zeros((N, K), dtype=np.int64)
        nbr_mask = np.zeros((N, K), dtype=bool)
        for i in range(N):
            nb = np.where(gn[i])[0]
            if len(nb) > K:
                nb = nb[:K]          # 截断（graph_neigh 已是局部邻居，顺序无偏好）
            nbr_idx[i, :len(nb)]  = nb
            nbr_mask[i, :len(nb)] = True
        self._refine_nbr_idx  = torch.as_tensor(nbr_idx,  device=self.device)
        self._refine_nbr_mask = torch.as_tensor(nbr_mask, device=self.device)

    def train(self):
        if self.datatype in ['Stereo', 'Slide']:
           self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
           self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
    
        # [REFINE] 构建精炼头 + 把它的参数加进优化器（否则不会被更新）
        if self.use_refine:
            self._build_refine_neighbors()
            self.refine_head = RefinementHead(
                dim=self.dim_output, attn_dim=self.refine_attn_dim
            ).to(self.device)
            _refine_lr = self.refine_lr if self.refine_lr is not None else self.learning_rate
            self.optimizer = torch.optim.Adam(
                [
                    {'params': self.model.parameters(),
                     'lr': self.learning_rate, 'weight_decay': self.weight_decay},
                    {'params': self.refine_head.parameters(),
                     'lr': _refine_lr, 'weight_decay': self.weight_decay},
                ]
            )
            print(f"[REFINE] RefinementHead enabled "
                  f"(attn_dim={self.refine_attn_dim}, n_nbr={self.refine_n_nbr}, "
                  f"lr={_refine_lr}); identity-init (alpha=0).")
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                              weight_decay=self.weight_decay)
        
        print('Begin to train ST data...')
        self.model.train()

        hard_neg_idx = None
        if 'hard_neg_idx' in self.adata.obsm.keys():
           hard_neg_idx = self.adata.obsm['hard_neg_idx']

        # [NEG] 负样本采样的可复现 generator
        _neg_rng = torch.Generator(device=self.device)
        _neg_rng.manual_seed(self.random_seed)
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()

            if hard_neg_idx is not None:
               hard_neg_idx = np.asarray(hard_neg_idx, dtype=np.int64)
               rand_idx = np.random.permutation(self.n_spot)
               mask = hard_neg_idx >= 0
               rand_idx[mask] = hard_neg_idx[mask]
               rand_idx = torch.as_tensor(rand_idx, dtype=torch.long, device=self.device)
               self.features_a = self.features.index_select(0, rand_idx)
            else:
                self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj)

            # [REFINE] 精炼 latent。identity-init 保证初期 refined == hiden_feat。
            #          精炼后的表示用于 NEG/PLR 损失与最终评测；模型表征学习信号
            #          会通过它反传到精炼头与 encoder。
            if self.use_refine:
                self.refine_head.train()
                self.refined_feat = self.refine_head(
                    self.hiden_feat, self._refine_nbr_idx, self._refine_nbr_mask
                )
            else:
                self.refined_feat = self.hiden_feat
            
            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features, self.emb)
            
            loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2)

            # ============ [NEG] 方案1：debiased node-to-node 对比 ============
            if self.gamma_neg > 0.0 and epoch >= self.neg_start_epoch:
                self._neg_epoch = epoch     # [NEG-S3] 供损失判断是否启用语义正样本
                # 按需刷新伪标签（仅当启用 pseudo debias）
                if self.neg_debias_pseudo:
                    if (self._neg_pseudo_labels is None or
                            epoch % self.neg_pseudo_refresh == 0):
                        self._refresh_pseudo_labels(self.refined_feat)
                loss_neg = self._refined_contrastive_loss(self.refined_feat, _neg_rng)
                loss = loss + self.gamma_neg * loss_neg
            # ================================================================

            # ============ [PLR] Pseudo-label Refinement（抗语义偏移）============
            if self.gamma_plr > 0.0 and epoch >= self.plr_start_epoch:
                if (self._plr_labels is None or
                        epoch % self.plr_refresh == 0):
                    self._refresh_plr(self.refined_feat)
                loss_plr = self._plr_loss(self.refined_feat)
                loss = loss + self.gamma_plr * loss_plr
            # ================================================================
            
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Optimization finished for ST data!")
        
        with torch.no_grad():
             self.model.eval()
             if self.deconvolution:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                
                return self.emb_rec
             else:  
                # [REFINE] 开启精炼头时：评测用"精炼后的 latent"（refine 作用的对象）；
                #          关闭时：严格走原路径（评测用 encoder 的 emb，index[1]），
                #          保证 use_refine=False 时与原版逐位等价。
                if self.use_refine:
                    self.refine_head.eval()
                    hf = self.model(self.features, self.features_a, self.adj)[0]  # hiden_feat
                    refined = self.refine_head(hf, self._refine_nbr_idx, self._refine_nbr_mask)
                    if self.datatype in ['Stereo', 'Slide']:
                        refined = F.normalize(refined, p=2, dim=1)
                    self.emb_rec = refined.detach().cpu().numpy()
                    self.adata.obsm['emb'] = self.emb_rec
                    return self.adata
                if self.datatype in ['Stereo', 'Slide']:
                   self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                   self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy() 
                else:
                   self.emb_rec = self.model(self.features, self.features_a, self.adj)[1].detach().cpu().numpy()
                self.adata.obsm['emb'] = self.emb_rec
                
                return self.adata
         
    def train_sc(self):
        self.model_sc = Encoder_sc(self.dim_input, self.dim_output).to(self.device)
        self.optimizer_sc = torch.optim.Adam(self.model_sc.parameters(), lr=self.learning_rate_sc)  
        
        print('Begin to train scRNA data...')
        for epoch in tqdm(range(self.epochs)):
            self.model_sc.train()
            
            emb = self.model_sc(self.feat_sc)
            loss = F.mse_loss(emb, self.feat_sc)
            
            self.optimizer_sc.zero_grad()
            loss.backward()
            self.optimizer_sc.step()
            
        print("Optimization finished for cell representation learning!")
        
        with torch.no_grad():
            self.model_sc.eval()
            emb_sc = self.model_sc(self.feat_sc)
         
            return emb_sc
        
    def train_map(self):
        emb_sp = self.train()
        emb_sc = self.train_sc()
        
        self.adata.obsm['emb_sp'] = emb_sp.detach().cpu().numpy()
        self.adata_sc.obsm['emb_sc'] = emb_sc.detach().cpu().numpy()
        
        # Normalize features for consistence between ST and scRNA-seq
        emb_sp = F.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc = F.normalize(emb_sc, p=2, eps=1e-12, dim=1)
        
        self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)  
          
        self.optimizer_map = torch.optim.Adam(self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        print('Begin to learn mapping matrix...')
        for epoch in tqdm(range(self.epochs)):
            self.model_map.train()
            self.map_matrix = self.model_map()

            loss_recon, loss_NCE = self.loss(emb_sp, emb_sc)
             
            loss = self.lamda1*loss_recon + self.lamda2*loss_NCE 

            self.optimizer_map.zero_grad()
            loss.backward()
            self.optimizer_map.step()
            
        print("Mapping matrix learning finished!")
        
        # take final softmax w/o computing gradients
        with torch.no_grad():
            self.model_map.eval()
            emb_sp = emb_sp.cpu().numpy()
            emb_sc = emb_sc.cpu().numpy()
            map_matrix = F.softmax(self.map_matrix, dim=1).cpu().numpy() # dim=1: normalization by cell
            
            self.adata.obsm['emb_sp'] = emb_sp
            self.adata_sc.obsm['emb_sc'] = emb_sc
            self.adata.obsm['map_matrix'] = map_matrix.T # spot x cell

            return self.adata, self.adata_sc
    
    def loss(self, emb_sp, emb_sc):
        # cell-to-spot
        map_probs = F.softmax(self.map_matrix, dim=1)   # dim=0: normalization by cell
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
           
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
           
        return loss_recon, loss_NCE
        
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M