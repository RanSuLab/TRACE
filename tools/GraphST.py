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

class RefinementHead(nn.Module):

    def __init__(self, dim: int, attn_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.attn_dim = attn_dim
        self.q = nn.Linear(dim, attn_dim, bias=False)
        self.k = nn.Linear(dim, attn_dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.alpha = nn.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, z, nbr_idx, nbr_mask):

        N, K = nbr_idx.shape
        q = self.q(z)
        k_all = self.k(z)
        v_all = self.v(z)

        k_nbr = k_all[nbr_idx]
        v_nbr = v_all[nbr_idx]

        scores = (q.unsqueeze(1) * k_nbr).sum(-1) / (self.attn_dim ** 0.5)
        scores = scores.masked_fill(~nbr_mask, float('-inf'))
        attn = torch.softmax(scores, dim=1)

        attn = torch.nan_to_num(attn, nan=0.0)

        delta = (attn.unsqueeze(-1) * v_nbr).sum(1)

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

        gamma_neg = 0.0,
        neg_temperature = 0.5,
        neg_n_neg = 256,
        neg_debias_spatial = True,
        neg_debias_pseudo = False,
        neg_pseudo_k = 7,
        neg_pseudo_refresh = 50,
        neg_start_epoch = 0,

        neg_semantic_pos = False,
        neg_sem_pos_k = 10,
        neg_sem_pos_start = 200,

        gamma_plr = 0.0,
        plr_k = 7,
        plr_start_epoch = 300,
        plr_refresh = 20,
        plr_conf_quantile = 0.5,
        plr_spatial_agree = 0.5,
        plr_temp = 1.0,

        use_refine = False,
        refine_attn_dim = 32,
        refine_n_nbr = 9,
        refine_lr = None,
        ):

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

        self.gamma_neg          = gamma_neg
        self.neg_temperature    = neg_temperature
        self.neg_n_neg          = neg_n_neg
        self.neg_debias_spatial = neg_debias_spatial
        self.neg_debias_pseudo  = neg_debias_pseudo
        self.neg_pseudo_k       = neg_pseudo_k
        self.neg_pseudo_refresh = neg_pseudo_refresh
        self.neg_start_epoch    = neg_start_epoch

        self.neg_semantic_pos   = neg_semantic_pos
        self.neg_sem_pos_k      = neg_sem_pos_k
        self.neg_sem_pos_start  = neg_sem_pos_start
        self._neg_epoch         = 0
        self._neg_pseudo_labels = None

        self.gamma_plr          = gamma_plr
        self.plr_k              = plr_k
        self.plr_start_epoch    = plr_start_epoch
        self.plr_refresh        = plr_refresh
        self.plr_conf_quantile  = plr_conf_quantile
        self.plr_spatial_agree  = plr_spatial_agree
        self.plr_temp           = plr_temp
        self._plr_centroids     = None
        self._plr_labels        = None
        self._plr_trusted       = None
        self._plr_prev_labels   = None

        self.use_refine         = use_refine
        self.refine_attn_dim    = refine_attn_dim
        self.refine_n_nbr       = refine_n_nbr
        self.refine_lr          = refine_lr
        self._refine_nbr_idx    = None
        self._refine_nbr_mask   = None
        self.refine_head        = None

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

           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else:

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

           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values

           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)

           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1]

           self.n_cell = adata_sc.n_obs
        self.n_spot = adata.n_obs

    def _refresh_pseudo_labels(self, hiden_feat):

        from sklearn.cluster import KMeans
        z = F.normalize(hiden_feat, dim=1).detach().cpu().numpy()
        km = KMeans(n_clusters=self.neg_pseudo_k, n_init=10,
                    random_state=self.random_seed).fit(z)
        self._neg_pseudo_labels = torch.as_tensor(
            km.labels_, dtype=torch.long, device=self.device
        )

    def _refined_contrastive_loss(self, hiden_feat, neg_rng):

        z = F.normalize(hiden_feat, dim=1)
        N = z.size(0)
        idx = torch.arange(N, device=self.device)
        sim_all = z @ z.T / self.neg_temperature

        gn = self.graph_neigh
        pos_mask = (gn > 0).clone()
        pos_mask[idx, idx] = False

        if (self.neg_semantic_pos and
                self._neg_epoch >= self.neg_sem_pos_start):
            k_sem = min(self.neg_sem_pos_k, N - 1)

            with torch.no_grad():
                sem_sim = sim_all.clone()
                sem_sim[idx, idx] = float('-inf')
                sem_topk = sem_sim.topk(k_sem, dim=1).indices
            sem_pos = torch.zeros_like(pos_mask)
            sem_pos[idx.unsqueeze(1), sem_topk] = True
            pos_mask = pos_mask | sem_pos

        has_pos = pos_mask.sum(1) > 0

        exclude = torch.zeros((N, N), dtype=torch.bool, device=self.device)
        if self.neg_debias_spatial:
            exclude = exclude | (gn > 0)
        exclude[idx, idx] = True
        if self.neg_debias_pseudo and self._neg_pseudo_labels is not None:
            pl = self._neg_pseudo_labels
            exclude = exclude | (pl.unsqueeze(0) == pl.unsqueeze(1))

        exclude = exclude | pos_mask

        n_neg = min(self.neg_n_neg, N - 1)
        neg_idx = torch.randint(0, N, (N, n_neg), device=self.device, generator=neg_rng)
        is_fake = exclude[idx.unsqueeze(1), neg_idx]
        neg_sim = torch.gather(sim_all, 1, neg_idx)
        neg_sim = neg_sim.masked_fill(is_fake, float('-inf'))
        neg_lse = torch.logsumexp(neg_sim, dim=1, keepdim=True)

        pos_sim = sim_all.masked_fill(~pos_mask, float('-inf'))
        pos_lse = torch.logsumexp(pos_sim, dim=1, keepdim=True)

        denom = torch.logaddexp(pos_lse, neg_lse)
        per_node = -(pos_lse - denom).squeeze(1)
        valid = has_pos & torch.isfinite(per_node)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return per_node[valid].mean()

    @torch.no_grad()
    def _refresh_plr(self, hiden_feat):

        from sklearn.cluster import KMeans
        z = F.normalize(hiden_feat, dim=1)
        z_np = z.detach().cpu().numpy()

        km = KMeans(n_clusters=self.plr_k, n_init=10,
                    random_state=self.random_seed).fit(z_np)
        labels = torch.as_tensor(km.labels_, dtype=torch.long, device=self.device)
        centroids = F.normalize(
            torch.as_tensor(km.cluster_centers_, dtype=torch.float32,
                            device=self.device), dim=1
        )

        sim_to_centroid = (z * centroids[labels]).sum(1)
        conf_thresh = torch.quantile(sim_to_centroid, self.plr_conf_quantile)
        conf_ok = sim_to_centroid >= conf_thresh

        gn = (self.graph_neigh > 0).float()
        idx = torch.arange(z.size(0), device=self.device)
        gn = gn.clone(); gn[idx, idx] = 0.0

        onehot = F.one_hot(labels, num_classes=self.plr_k).float()
        nbr_label_count = gn @ onehot
        nbr_total = gn.sum(1).clamp_min(1.0)
        agree = nbr_label_count[idx, labels] / nbr_total
        spatial_ok = (gn.sum(1) > 0) & (agree >= self.plr_spatial_agree)

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

        if self._plr_centroids is None or self._plr_trusted is None:
            return torch.tensor(0.0, device=self.device)
        z = F.normalize(hiden_feat, dim=1)
        logits = (z @ self._plr_centroids.T) / self.plr_temp
        ce = F.cross_entropy(logits, self._plr_labels, reduction='none')
        m = self._plr_trusted.float()
        if m.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return (ce * m).sum() / m.sum()

    def _build_refine_neighbors(self):

        gn = (self.graph_neigh > 0).detach().cpu().numpy()
        np.fill_diagonal(gn, False)
        N = gn.shape[0]
        K = self.refine_n_nbr
        nbr_idx  = np.zeros((N, K), dtype=np.int64)
        nbr_mask = np.zeros((N, K), dtype=bool)
        for i in range(N):
            nb = np.where(gn[i])[0]
            if len(nb) > K:
                nb = nb[:K]
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

            if self.gamma_neg > 0.0 and epoch >= self.neg_start_epoch:
                self._neg_epoch = epoch

                if self.neg_debias_pseudo:
                    if (self._neg_pseudo_labels is None or
                            epoch % self.neg_pseudo_refresh == 0):
                        self._refresh_pseudo_labels(self.refined_feat)
                loss_neg = self._refined_contrastive_loss(self.refined_feat, _neg_rng)
                loss = loss + self.gamma_neg * loss_neg

            if self.gamma_plr > 0.0 and epoch >= self.plr_start_epoch:
                if (self._plr_labels is None or
                        epoch % self.plr_refresh == 0):
                    self._refresh_plr(self.refined_feat)
                loss_plr = self._plr_loss(self.refined_feat)
                loss = loss + self.gamma_plr * loss_plr

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

                if self.use_refine:
                    self.refine_head.eval()
                    hf = self.model(self.features, self.features_a, self.adj)[0]
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

        with torch.no_grad():
            self.model_map.eval()
            emb_sp = emb_sp.cpu().numpy()
            emb_sc = emb_sc.cpu().numpy()
            map_matrix = F.softmax(self.map_matrix, dim=1).cpu().numpy()

            self.adata.obsm['emb_sp'] = emb_sp
            self.adata_sc.obsm['emb_sc'] = emb_sc
            self.adata.obsm['map_matrix'] = map_matrix.T

            return self.adata, self.adata_sc

    def loss(self, emb_sp, emb_sc):

        map_probs = F.softmax(self.map_matrix, dim=1)
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)

        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)

        return loss_recon, loss_NCE

    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        mat = self.cosine_similarity(pred_sp, emb_sp)
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))

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
