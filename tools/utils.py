import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.decomposition import PCA
import rpy2.robjects as robjects

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    np.random.seed(random_seed)
    robjects.r.library("mclust")

    emb = np.array(adata.obsm[used_obsm], dtype=np.float64, order='F')
    n_row, n_col = emb.shape
    print(emb.shape)

    robjects.r.assign(
        'mclust_input',
        robjects.r['matrix'](
            robjects.FloatVector(emb.flatten(order='F')),
            nrow=n_row,
            ncol=n_col,
        )
    )

    robjects.r(f'set.seed({random_seed})')
    robjects.r(f'mclust_res <- Mclust(mclust_input, G={num_cluster}, modelNames="{modelNames}")')
    mclust_res = np.array(robjects.r('mclust_res$classification'))

    robjects.r('rm(mclust_input, mclust_res)')

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False, n_components=20):

    pca = PCA(n_components=n_components, random_state=42)
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding

    if method == 'mclust':
       adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain']

    if refinement:
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    old_type = adata.obs[key].values
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]

    new_type = []
    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = [old_type[index[j]] for j in range(1, n_neigh+1)]
        new_type.append(max(neigh_type, key=neigh_type.count))
    new_type = np.array(new_type)

    import collections
    orig_labels = set(old_type)
    refined_labels = set(new_type)
    lost = orig_labels - refined_labels

    for lab in lost:

        members = np.where(old_type == lab)[0]
        if len(members) == 0:
            continue

        keep = []
        for i in members:
            index = distance[i, :].argsort()
            neigh = [old_type[index[j]] for j in range(1, n_neigh+1)]
            same_ratio = neigh.count(lab) / n_neigh
            keep.append((i, same_ratio))

        keep.sort(key=lambda x: x[1], reverse=True)
        n_restore = max(1, len(keep) // 2)
        for i, _ in keep[:n_restore]:
            new_type[i] = lab

    return [str(i) for i in new_type]

def extract_top_value(map_matrix, retain_percent = 0.1):

    top_k  = retain_percent * map_matrix.shape[1]
    output = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)

    return output

def construct_cell_type_matrix(adata_sc):
    label = 'cell_type'
    n_type = len(list(adata_sc.obs[label].unique()))
    zeros = np.zeros([adata_sc.n_obs, n_type])
    cell_type = list(adata_sc.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    mat = pd.DataFrame(zeros, index=adata_sc.obs_names, columns=cell_type)
    for cell in list(adata_sc.obs_names):
        ctype = adata_sc.obs.loc[cell, label]
        mat.loc[cell, str(ctype)] = 1

    return mat

def project_cell_to_spot(adata, adata_sc, retain_percent=0.1):

    map_matrix = adata.obsm['map_matrix']

    map_matrix = extract_top_value(map_matrix)

    matrix_cell_type = construct_cell_type_matrix(adata_sc)
    matrix_cell_type = matrix_cell_type.values

    matrix_projection = map_matrix.dot(matrix_cell_type)

    cell_type = list(adata_sc.obs['cell_type'].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()

    df_projection = pd.DataFrame(matrix_projection, index=adata.obs_names, columns=cell_type)

    df_projection = df_projection.div(df_projection.sum(axis=1), axis=0).fillna(0)

    adata.obs[df_projection.columns] = df_projection

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):

    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!."

    return res
