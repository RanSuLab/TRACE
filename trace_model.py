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

    def __init__(self, edge_index: np.ndarray, ema_decay: float = 0.7):
        self.edge_index = edge_index
        self.n_edges    = edge_index.shape[1]
        self.ema_decay  = ema_decay

        self.ema_residual = np.full(self.n_edges, 0.5, dtype=np.float32)
        self._initialized = False

    @torch.no_grad()
    def update(self, edge_logits: torch.Tensor,
               edge_indices: Optional[np.ndarray] = None) -> None:

        probs     = torch.sigmoid(edge_logits).cpu().numpy().astype(np.float32)
        residuals = np.abs(1.0 - probs)

        self._initialized = True

        if edge_indices is None:

            if len(residuals) == self.n_edges:
                self.ema_residual = (
                    self.ema_decay * self.ema_residual
                    + (1.0 - self.ema_decay) * residuals
                )
        else:

            idx = np.asarray(edge_indices).ravel().astype(np.int64)
            valid = (idx >= 0) & (idx < self.n_edges)
            idx_v = idx[valid]
            res_v = residuals[valid]
            self.ema_residual[idx_v] = (
                self.ema_decay * self.ema_residual[idx_v]
                + (1.0 - self.ema_decay) * res_v
            )

    def get_difficulty(self) -> np.ndarray:

        mn, mx = self.ema_residual.min(), self.ema_residual.max()
        return ((self.ema_residual - mn) / (mx - mn + 1e-8)).astype(np.float32)

    def get_spot_difficulty(self, n_spots: int) -> np.ndarray:
        diff   = self.get_difficulty()
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

    def get_spot_confidence(self, n_spots: int) -> np.ndarray:

        return 1.0 - self.get_spot_difficulty(n_spots)

class CrossCorrelationDecoder(nn.Module):

    def __init__(self, embed_dim: int, n_layers: int = 1, hidden_dim: int = 128):
        super().__init__()

        in_dim = (n_layers ** 2) * embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_list: List[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        src, dst = edge_index[0], edge_index[1]

        parts = []
        for hk in h_list:
            for hj in h_list:

                parts.append(hk[src] * hj[dst])

        h_edge = torch.cat(parts, dim=-1)

        logits = self.mlp(h_edge).squeeze(-1)
        return logits

class PrototypeContrastiveLoss(nn.Module):

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

    def _soft_assignment(self, z: torch.Tensor) -> torch.Tensor:

        z_norm    = F.normalize(z, dim=1)
        proto     = F.normalize(self.prototypes, dim=1)
        logits    = z_norm @ proto.T / self.temperature
        return F.softmax(logits, dim=1)

    @staticmethod
    def _sharpen(q: torch.Tensor) -> torch.Tensor:

        q2   = q ** 2
        f    = q2.sum(dim=0, keepdim=True)
        p    = q2 / (f + 1e-8)
        return p / (p.sum(dim=1, keepdim=True) + 1e-8)

    def forward(
        self,
        z: torch.Tensor,
        spatial_nbr_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        q = self._soft_assignment(z)

        with torch.no_grad():
            p = self._sharpen(q)

        loss_compact = F.kl_div(
            (q + 1e-8).log(), p, reduction='batchmean'
        )

        mean_q       = q.mean(dim=0)
        loss_uniform = (mean_q * (mean_q + 1e-8).log()).sum()

        loss = loss_compact + self.uniform_weight * loss_uniform

        if spatial_nbr_idx is not None and self.spatial_weight > 0.0:

            N, k   = spatial_nbr_idx.shape
            nbr_q  = q[spatial_nbr_idx.reshape(-1)].reshape(N, k, self.n_clusters)

            nbr_mean = nbr_q.mean(dim=1)
            nbr_mean = (nbr_mean / (nbr_mean.sum(dim=1, keepdim=True) + 1e-8))

            loss_spatial = F.kl_div(
                (q + 1e-8).log(), nbr_mean.detach(), reduction='batchmean'
            )
            loss = loss + self.spatial_weight * loss_spatial

        return loss

class CurriculumEdgeMasker:

    def __init__(
        self,
        n_edges: int,
        mask_ratio: float = 0.5,
        split_ratio: float = 0.5,
    ):
        self.n_edges    = n_edges
        self.mask_ratio = mask_ratio
        self.split_ratio = split_ratio

        self._lambda = 0.0

    def update_lambda(self, epoch: int, total_epochs: int) -> None:

        warmup_end = int(total_epochs * (2 / 3))
        if epoch < warmup_end:
            self._lambda = epoch / max(warmup_end, 1)
        else:
            self._lambda = 1.0

    def select_mask_edges(
        self,
        difficulty: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:

        if rng is None:
            rng = np.random.default_rng(42)

        K_target = int(np.ceil(self._lambda * self.mask_ratio * self.n_edges))
        K_target = min(K_target, int(self.mask_ratio * self.n_edges))
        K_target = max(K_target, 1)

        order_easy_first = np.argsort(difficulty)

        n_guided = max(1, int(np.ceil(self.split_ratio * K_target)))
        guided_idx = order_easy_first[:n_guided]

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

class SpotConfidenceEstimator:

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

class SelfPacedCurriculum:

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

        self.n_spots             = spatial_coords.shape[0]
        self.n_spatial_neighbors = n_spatial_neighbors
        self.device              = device or torch.device('cpu')
        self.spatial_adj         = self._build_knn_adj(spatial_coords,
                                                        n_spatial_neighbors)
        self.image_adj           = None
        self._spatial_neighbors  = self._adj_to_neighbor_list(self.spatial_adj)
        self._image_neighbors    = None
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

    def build_adj(self, confidence, mask_m, n_extra, max_img_dist=None):

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

    import scipy.sparse as sp
    if sp.issparse(adj):
        coo = adj.tocoo()
        rows = coo.row.astype(np.int64)
        cols = coo.col.astype(np.int64)

        mask = np.asarray(coo.data).ravel() > 0
        rows, cols = rows[mask], cols[mask]
    else:
        adj_arr = np.asarray(adj)
        rows, cols = np.where(adj_arr > 0)
        rows = rows.astype(np.int64)
        cols = cols.astype(np.int64)
    return np.stack([rows, cols], axis=0)

def load_img_embs_h5(
    h5_path: str,
    barcode_order=None,
) -> np.ndarray:

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"[load_img_embs_h5] img_embs.h5 not found: {h5_path}\n"
            "  请先运行特征提取脚本 extract_uni_features.py 生成该文件。"
        )

    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:]
        raw_barcodes = f["barcodes"][:]

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

class AdaMixGraphST:

    def __init__(
        self,
        adata,
        image_patches: Optional[np.ndarray] = None,
        image_embeddings: Optional[np.ndarray] = None,
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
        gamma_cluster: float = 0.0,
        proto_temperature: float = 0.2,
        proto_uniform_weight: float = 0.1,
        proto_spatial_weight: float = 0.5,
        dim_proto = 3000,

        use_latent_semantic: bool = True,
        latent_sem_k: int = 6,
        latent_sem_refresh: int = 50,
        latent_sem_strength: float = 0.30,
        latent_sem_start_epoch: Optional[int] = None,

        gamma_smooth: float = 0.0,
        smooth_start_epoch: int = 0,

        curriculum_mode: str = "easy_to_hard",
        record_diagnostics: bool = True,
        respect_input_adj: bool = False,
        use_feature_perturbation: bool = True,
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

        self.use_latent_semantic   = use_latent_semantic
        self.latent_sem_k          = latent_sem_k
        self.latent_sem_refresh    = latent_sem_refresh
        self.latent_sem_strength   = latent_sem_strength
        self.latent_sem_start_epoch = (
            latent_sem_start_epoch if latent_sem_start_epoch is not None
            else warmup_epochs
        )

        self._latent_sem_adj_bin: Optional[np.ndarray] = None
        self._latent_sem_last_update: int = -1

        self.gamma_smooth       = gamma_smooth
        self.smooth_start_epoch = smooth_start_epoch

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

        self.curriculum    = SelfPacedCurriculum(max_iter=epochs, K_max=K_max)
        self.graph_builder = AdaptiveGraphBuilder(
            spatial_coords=self.adata.obsm['spatial'],
            n_spatial_neighbors=n_spatial_neighbors,
            n_image_neighbors=n_image_neighbors,
            device=self.device,
            top_fraction=self.top_fraction,
        )

        if self.respect_input_adj:
            self.graph_builder.spatial_adj = self._base_adj_binary_np.copy()
            self.graph_builder._spatial_neighbors = self.graph_builder._adj_to_neighbor_list(
                self.graph_builder.spatial_adj
            )

        spatial_adj_np  = self.adata.obsm['adj']
        self._edge_index_np = adj_to_edge_index(spatial_adj_np)
        n_edges = self._edge_index_np.shape[1]

        self.edge_difficulty_estimator = EdgeDifficultyEstimator(
            edge_index=self._edge_index_np,
            ema_decay=ema_decay,
        )

        self.confidence_estimator = SpotConfidenceEstimator(
            n_spots, ema_decay=ema_decay, use_pearson=True
        )

        self.edge_masker = CurriculumEdgeMasker(
            n_edges=n_edges,
            mask_ratio=mask_ratio,
            split_ratio=split_ratio,
        )
        self._mask_ratio  = mask_ratio
        self._split_ratio = split_ratio

        self._edge_index_t = torch.tensor(
            self._edge_index_np, dtype=torch.long
        ).to(self.device)

        _nbrs_obj = NearestNeighbors(
            n_neighbors=n_spatial_neighbors + 1, algorithm='auto'
        ).fit(self.adata.obsm['spatial'])
        _, _nbr_ind = _nbrs_obj.kneighbors(self.adata.obsm['spatial'])
        self._spatial_nbr_idx = _nbr_ind[:, 1:].astype(np.int64)

        if image_embeddings is not None or image_patches is not None:
            print("[AdaMixGraphST] [UNI-REMOVED] image inputs are ignored "
                  "(gene-expression-only mode).")

        self._cc_hidden = cc_hidden

        self._hard_neg_ei = self._build_2hop_neg_pool()

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

        while len(neg_src) < n_neg:
            neg_src.append(neg_src[-1])
            neg_dst.append(neg_dst[-1])
        neg_ei = np.stack([neg_src[:n_neg], neg_dst[:n_neg]], axis=0)
        return torch.tensor(neg_ei, dtype=torch.long).to(self.device)

    def _get_adaptive_adj(self, confidence, mask_m, n_extra):

        adj_np = self.graph_builder.build_adj(confidence, mask_m, n_extra, max_img_dist=self.max_img_dist)
        adj_norm = normalize_adj(adj_np)
        return torch.FloatTensor(adj_norm).to(self.device)

    def _blend_graphs(self, adaptive_adj, graph_strength):
        graph_strength = float(np.clip(graph_strength, 0.0, 1.0))
        return (1.0 - graph_strength) * self._base_adj + graph_strength * adaptive_adj

    @staticmethod
    def _laplacian_smoothness_loss(
        hiden_feat: torch.Tensor,
        spatial_nbr_idx: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:

        z = hiden_feat
        if normalize:
            z = F.normalize(z, dim=1)

        z_nbr = z[spatial_nbr_idx]

        diff = z.unsqueeze(1) - z_nbr

        loss = diff.pow(2).sum(dim=-1).mean()
        return loss

    @torch.no_grad()
    def _update_latent_semantic_adj(
        self,
        hiden_feat: torch.Tensor,
        epoch: int,
    ) -> None:

        z = F.normalize(hiden_feat, dim=1).detach().cpu().numpy()
        n = z.shape[0]
        k = min(self.latent_sem_k, n - 1)

        sim = z @ z.T
        np.fill_diagonal(sim, -np.inf)

        top_idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]

        adj = np.zeros((n, n), dtype=np.float32)
        rows = np.repeat(np.arange(n), k)
        cols = top_idx.reshape(-1)
        adj[rows, cols] = 1.0
        adj = np.clip(adj + adj.T, 0.0, 1.0)

        self._latent_sem_adj_bin    = adj
        self._latent_sem_last_update = epoch

    def _should_refresh_latent_sem(self, epoch: int) -> bool:

        if not self.use_latent_semantic:
            return False
        if epoch < self.latent_sem_start_epoch:
            return False

        if self._latent_sem_last_update < 0:
            return True
        return (epoch - self._latent_sem_last_update) >= self.latent_sem_refresh

    def _get_adaptive_adj_binary(self, confidence, mask_m, n_extra) -> np.ndarray:

        return self.graph_builder.build_adj(
            confidence, mask_m, n_extra, max_img_dist=self.max_img_dist
        ).astype(np.float32)

    def _build_train_adj(
        self,
        adaptive_adj_bin: np.ndarray,
        graph_strength: float,
        masked_edge_idx: Optional[np.ndarray] = None,
        latent_sem_adj_bin: Optional[np.ndarray] = None,
        latent_sem_strength: float = 0.0,
    ) -> torch.Tensor:

        gs = float(np.clip(graph_strength, 0.0, 1.0))
        ss = float(np.clip(latent_sem_strength, 0.0, 1.0))

        total_external = gs + ss
        if total_external > 1.0:
            gs = gs / total_external
            ss = ss / total_external
            total_external = 1.0

        base_w = 1.0 - total_external

        adj_blend = base_w * self._base_adj_binary_np + gs * adaptive_adj_bin
        if latent_sem_adj_bin is not None and ss > 0.0:
            adj_blend = adj_blend + ss * latent_sem_adj_bin

        if masked_edge_idx is not None and len(masked_edge_idx) > 0:
            src = self._edge_index_np[0][masked_edge_idx]
            dst = self._edge_index_np[1][masked_edge_idx]
            adj_blend[src, dst] = 0.0
            adj_blend[dst, src] = 0.0

        adj_norm = normalize_adj(adj_blend)
        return torch.FloatTensor(adj_norm).to(self.device)

    def _apply_edge_mask(
        self,
        adj: torch.Tensor,
        masked_edge_idx: np.ndarray,
    ) -> torch.Tensor:

        adj_masked = adj.clone()
        src = self._edge_index_np[0][masked_edge_idx]
        dst = self._edge_index_np[1][masked_edge_idx]
        adj_masked[src, dst] = 0.0
        adj_masked[dst, src] = 0.0
        return adj_masked

    def _select_experiment_mask_edges(self, difficulty, rng, epoch):
        mode = self.curriculum_mode
        if mode == "none":
            return np.empty(0, dtype=np.int64)

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

            return np.argsort(-difficulty)[:K_target].astype(np.int64)

        return self.edge_masker.select_mask_edges(difficulty=difficulty, rng=rng)

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

    def train(self) -> object:

        from GraphST.model import Encoder
        from tqdm import tqdm

        model = Encoder(
            self.dim_input, self.dim_output, self.graph_neigh
        ).to(self.device)
        loss_CSL = nn.BCEWithLogitsLoss()
        self.topo_reg_fn = GraphTopologyRegularization(align_weight=1.0, sparse_weight=0.01).to(self.device)

        cc_decoder = CrossCorrelationDecoder(
            embed_dim=self.dim_output,
            n_layers=1,
            hidden_dim=self._cc_hidden,
        ).to(self.device)

        proto_loss_fn = PrototypeContrastiveLoss(
            n_clusters=self._n_clusters,
            embed_dim=self.dim_proto,
            temperature=self._proto_temperature,
            uniform_weight=self._proto_uniform_w,
            spatial_weight=self._proto_spatial_w,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(cc_decoder.parameters())
            + list(proto_loss_fn.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        print("== AdaMixGraphST (+ Cur-MGAE) training started ==")

        features_a    = self.features
        final_adj_used = self._base_adj
        rng = np.random.default_rng(42)

        _spatial_nbr_t = torch.tensor(
            self._spatial_nbr_idx, dtype=torch.long
        ).to(self.device)

        for epoch in tqdm(range(self.epochs)):
            model.train()
            cc_decoder.train()
            proto_loss_fn.train()

            lam = self.curriculum.age_param(epoch)

            old_conf = self.confidence_estimator.get_confidence()
            edge_conf = self.edge_difficulty_estimator.get_spot_confidence(self.n_spots)

            spot_conf = edge_conf
            self.confidence_estimator.fused_confidence = spot_conf

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

                weight_v = min(raw_weight_v, max_strength)
                if max_strength > 0.0:
                    weight_v = max(weight_v, min_strength)

                direction_soft = 0.1
                mask_m = 1 if weight_v >= direction_soft else 0

                n_extra         = self.curriculum.n_perturbations(weight_v)
                perturb_fraction = min(0.30, 0.30 * weight_v)
                graph_strength  = min(0.30, 0.30 * weight_v)

            self.edge_masker.update_lambda(epoch, self.epochs)
            edge_difficulty = self.edge_difficulty_estimator.get_difficulty()
            masked_edge_idx = self._select_experiment_mask_edges(
                difficulty=edge_difficulty, rng=rng, epoch=epoch
            )

            adaptive_adj_bin = self._get_adaptive_adj_binary(
                spot_conf, mask_m, n_extra
            )

            cur_sem_strength = (
                self.latent_sem_strength
                if (self.use_latent_semantic
                    and epoch >= self.latent_sem_start_epoch
                    and self._latent_sem_adj_bin is not None)
                else 0.0
            )

            adj_masked = self._build_train_adj(
                adaptive_adj_bin,
                graph_strength,
                masked_edge_idx=masked_edge_idx,
                latent_sem_adj_bin=self._latent_sem_adj_bin,
                latent_sem_strength=cur_sem_strength,
            )

            adj_used = self._build_train_adj(
                adaptive_adj_bin,
                graph_strength,
                masked_edge_idx=None,
                latent_sem_adj_bin=self._latent_sem_adj_bin,
                latent_sem_strength=cur_sem_strength,
            )

            if self.use_feature_perturbation and perturb_fraction > 0.0:
                feat_a_np = self.graph_builder.build_perturbed_features(
                    features=self.features_np,
                    confidence=spot_conf,
                    mask_m=mask_m,
                    n_swaps=max(1, n_extra),
                    mix_ratio_easy=0.15 + 0.10 * weight_v,
                    mix_ratio_hard=0.30 + 0.20 * weight_v,

                    perturb_fraction=perturb_fraction,
                )
                features_a = torch.FloatTensor(feat_a_np).to(self.device)
            else:
                features_a = self.features

            hiden_feat, emb, ret, ret_a = model(
                self.features, features_a, adj_masked
            )

            if self._should_refresh_latent_sem(epoch):
                self._update_latent_semantic_adj(hiden_feat, epoch)

            if (epoch == self.warmup_epochs and not proto_loss_fn.initialized):
                proto_loss_fn.initialize_prototypes(emb.detach())

            loss_sl_1  = loss_CSL(ret,   self.label_CSL)
            loss_sl_2  = loss_CSL(ret_a, self.label_CSL)
            loss_recon = F.mse_loss(self.features, emb)

            h_list = [hiden_feat]
            loss_edge, pos_logits, masked_edge_idx = self._compute_edge_recon_loss(
                cc_decoder, h_list, masked_edge_idx
            )

            gamma_cluster_eff = (
                self.gamma_cluster if epoch >= self.warmup_epochs else 0.0
            )
            if gamma_cluster_eff > 0.0:
                loss_cluster = proto_loss_fn(

                    emb,
                    spatial_nbr_idx=_spatial_nbr_t,
                )
            else:
                loss_cluster = torch.tensor(0.0, device=self.device)

            gamma_smooth_eff = (
                self.gamma_smooth if epoch >= self.smooth_start_epoch else 0.0
            )
            if gamma_smooth_eff > 0.0:
                loss_smooth = self._laplacian_smoothness_loss(
                    hiden_feat, _spatial_nbr_t, normalize=True
                )
            else:
                loss_smooth = torch.tensor(0.0, device=self.device)

            loss = (
                self.alpha  * loss_recon
                + self.beta * (loss_sl_1 + loss_sl_2)
                + self.gamma_edge    * loss_edge
                + gamma_cluster_eff  * loss_cluster
                + gamma_smooth_eff   * loss_smooth
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                cc_decoder.eval()
                proto_loss_fn.eval()

                if len(masked_edge_idx) > 0:
                    self.edge_difficulty_estimator.update(
                        pos_logits, edge_indices=masked_edge_idx
                    )

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
                proto_loss_fn.train()

            final_adj_used = adj_used

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
                    f"loss_cluster={loss_cluster.item():.4f}, "
                    f"loss_smooth={loss_smooth.item():.4f}, "
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
