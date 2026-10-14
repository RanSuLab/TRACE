"""TRACE command-line utility."""
import argparse
import inspect
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from trace_model import TRACE

DEMO_DEFAULTS = {'K_max': 9,
 'alpha': 10.0,
 'beta': 1.0,
 'curriculum_mode': 'easy_to_hard',
 'dim_output': 64,
 'ema_decay': 0.4,
 'epochs': 1500,
 'gamma_cluster': 0.0,
 'gamma_edge': 2.0,
 'latent_sem_refresh': 50,
 'latent_sem_strength': 0.6,
 'n_image_neighbors': 3,
 'n_spatial_neighbors': 9,
 'random_seed': 41,
 'record_diagnostics': True,
 'use_latent_semantic': True,
 'warmup_epochs': 400}
DEMO_EVALUATION = {'method': 'mclust', 'radius': 50, 'refinement': True}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--metadata', type=Path)
    p.add_argument('--id-column', help='Metadata barcode column; default: first column')
    p.add_argument('--label-column', default='layer_guess')
    p.add_argument('--n-clusters', type=int, default=7)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.n_clusters < 2:
        p.error('--n-clusters must be >=2')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output directory is not empty; select a new directory.')
    data = sc.read_visium(str(a.data_dir), count_file='filtered_feature_bc_matrix.h5', load_images=False)
    data.var_names_make_unique()
    if not data.obs_names.is_unique:
        raise ValueError('Duplicate spot barcodes')
    data.obs['ground_truth'] = np.nan
    if a.metadata:
        if a.id_column:
            meta = pd.read_csv(a.metadata, sep='\t', dtype={a.id_column: str})
            if a.id_column not in meta:
                raise ValueError('Metadata is missing the requested barcode column')
            meta = meta.set_index(a.id_column)
        else:
            meta = pd.read_csv(a.metadata, sep='\t', index_col=0)
        meta.index = meta.index.astype(str)
        if not meta.index.is_unique or a.label_column not in meta:
            raise ValueError('Duplicate metadata IDs or missing annotation column')
        if not len(data.obs_names.intersection(meta.index)):
            raise ValueError('No matching metadata barcodes; row-order matching is not used')
        data.obs['ground_truth'] = meta[a.label_column].reindex(data.obs_names)
    kwargs = dict(DEMO_DEFAULTS, device=torch.device(a.device), n_clusters=a.n_clusters)
    allowed = inspect.signature(TRACE.__init__).parameters
    extra = set(kwargs) - set(allowed)
    if extra:
        raise ValueError('Selected model does not support demo arguments: ' + str(sorted(extra)))
    start = time.perf_counter()
    trained = TRACE(data, **kwargs).train()
    if trained is None or 'emb' not in trained.obsm:
        raise ValueError('Model did not return an embedding')
    z = np.asarray(trained.obsm['emb'])
    if z.ndim != 2 or len(z) != trained.n_obs or not np.isfinite(z).all():
        raise ValueError('Invalid embedding')
    from tools.utils import clustering
    clustering(trained, n_clusters=a.n_clusters, **DEMO_EVALUATION)
    xy = np.asarray(trained.obsm['spatial'])[:, :2]
    pred = pd.DataFrame({'spot_id': trained.obs_names.astype(str),
                         'spatial_x': xy[:, 0], 'spatial_y': xy[:, 1],
                         'ground_truth': trained.obs['ground_truth'].to_numpy(),
                         'Ours': trained.obs['domain'].astype(str).to_numpy()})
    a.output.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.output / 'Ours_predictions.csv', index=False)
    pd.DataFrame(z, index=trained.obs_names).to_csv(a.output / 'Ours_embedding.tsv', sep='\t', index_label='spot_id')
    valid = pred.ground_truth.notna() & pred.Ours.notna()
    result = {'n_spots': len(pred), 'n_eval_spots': int(valid.sum()),
              'n_domains_found': int(pred.Ours.nunique()), 'runtime_sec': time.perf_counter() - start}
    if valid.any():
        result['ARI'] = float(adjusted_rand_score(pred.loc[valid, 'ground_truth'].astype(str), pred.loc[valid, 'Ours']))
        result['NMI'] = float(normalized_mutual_info_score(pred.loc[valid, 'ground_truth'].astype(str), pred.loc[valid, 'Ours']))
    pd.DataFrame([result]).to_csv(a.output / 'metrics.csv', index=False)
    print(json.dumps(result, indent=2))
    print('Saved:', a.output.resolve())

if __name__ == '__main__':
    main()
