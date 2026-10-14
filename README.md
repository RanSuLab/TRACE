# TRACE

Training-guided Relational Adaptation for spatial Context Embedding.

## Scope

This release contains a model implementation, one fixed-default training demo
based on DLPFC 151673, and selected saved-result analysis scripts.
The training demo does not reproduce all dataset-specific manuscript runs.
Historical YAML configurations, run manifests and tuning logs are not included.
Legacy class/output names (`AdaMixGraphST`, `Ours`) are retained for compatibility.

## Environment

TODO: add and test `environment.yml` from the original training environment.
PyTorch, Scanpy, AnnData, NumPy, pandas, SciPy, scikit-learn, matplotlib,
h5py, tqdm and POT are required. GraphST utilities also import rpy2;
R and the mclust package are required for the default clustering path.
PyYAML is needed by the packaging script only, not by train_demo.py.

## Training demo

Download the DLPFC Visium data and provide barcode-indexed metadata.

```bash
python train_demo.py --data-dir Data/1.DLPFC/151673 \
  --metadata Data/1.DLPFC/151673/metadata.tsv \
  --label-column layer_guess --device cuda:0 --output Results/demo_151673
```

Metadata is optional and used only for evaluation, not for model training.
The default first metadata column must contain barcode IDs; alternatively
specify `--id-column`. Row-order annotation matching is not performed.
For another section, provide its directory and the correct `--n-clusters`.
The output directory must be new or empty.

## Saved-result analysis

Use the provided result directories with the analysis entry points.
Always pass explicit data/result and output paths. Raw expression and annotations
must be downloaded from their original sources. Domain IDs are specific to the
released predictions. Example commands are provided in docs/analysis_commands.md.
