# TRACE

**Training-guided Relational Adaptation for spatial Context Embedding**

TRACE is a self-supervised framework for spatial domain identification from gene expression and spatial coordinates. It combines progressive edge masking, confidence-guided expression perturbation, and dynamically updated spatial and latent semantic neighborhoods.

This repository provides a DLPFC training example with one fixed default configuration and an HBC analysis workflow using saved predictions.

## Repository contents

```text
TRACE/
├── trace_model.py
├── train_demo.py
├── GraphST/
├── analysis/
│   └── plot_hbc.py
├── docs/
│   ├── analysis_commands.md
│   └── data_sources.md
├── environment.yml
├── requirements.txt
├── r-session-info.txt
└── THIRD_PARTY_NOTICES.md
```

The implementation retains the class name `AdaMixGraphST` and prediction column `Ours` for compatibility with existing results. The manuscript method name is TRACE.

## Environment

The supplied environment export records the original Linux environment, including Python 3.8.20, PyTorch 2.4.1, Scanpy 1.9.1, AnnData 0.8.0, NumPy 1.22.3, pandas 1.4.2, SciPy 1.8.1, scikit-learn 1.1.1, matplotlib 3.4.2, and rpy2 3.4.1. The R session record reports R 4.1.2 on Ubuntu 22.04.5 LTS.

The default clustering workflow requires R and the R package `mclust`. R must be accessible to rpy2. The supplied R session record does not identify the installed mclust version.

### Installation

```bash
git clone https://github.com/RanSuLab/TRACE.git
cd TRACE
```

Before creating the environment:

1. Remove the machine-specific `prefix:` entry from `environment.yml`.
2. Make R available on the machine. R is not included in the supplied environment export.
3. Install `mclust` in the R library accessible from that R installation if it is not already installed.

```bash
R --version
Rscript -e 'if (!requireNamespace("mclust", quietly = TRUE)) install.packages("mclust", repos = "https://cloud.r-project.org")'
conda env create -f environment.yml
conda activate TRACE
```

Verify the R interface and Python imports:

```bash
python -c "import torch, scanpy, anndata, numpy, pandas, scipy, sklearn, matplotlib, h5py, ot; from rpy2 import robjects; robjects.r.library('mclust'); from trace_model import AdaMixGraphST; print('Imports OK')"
python train_demo.py --help
python analysis/plot_hbc.py --help
```

`requirements.txt` is an additional environment record. Its current export contains local build paths and should not be used directly with `pip install -r requirements.txt`. The environment export also contains packages beyond the dependencies of this minimal release. A successful clean installation and complete training run should be verified on the target machine.

## DLPFC training example

The example uses fixed defaults based on DLPFC section **151673**. Training does not read YAML files. It illustrates the TRACE workflow; it is not an entry point for reproducing every dataset-specific manuscript training run.

### Input files

```text
Data/1.DLPFC/151673/
├── filtered_feature_bc_matrix.h5
├── metadata.tsv
└── spatial/
    ├── tissue_positions_list.csv
    └── ...
```

Use the Visium count matrix and corresponding spatial files. The training example uses `scanpy.read_visium(..., load_images=False)` and does not require image embeddings. Obtain the DLPFC expression data and cortical-layer annotations from the original dataset release associated with:

> Maynard et al. Transcriptome-scale spatial gene expression in the human dorsolateral prefrontal cortex. Nature Neuroscience (2021).

The optional metadata file is tab-separated. Its first column must contain unique spot barcodes, and `layer_guess` must contain the evaluation labels. If barcodes use another column, specify `--id-column`. Metadata is matched by barcode, not by row order.

### Run

```bash
python train_demo.py \
  --data-dir Data/1.DLPFC/151673 \
  --metadata Data/1.DLPFC/151673/metadata.tsv \
  --label-column layer_guess \
  --device cuda:0 \
  --output Results/demo_151673
```

For metadata with a named barcode column:

```bash
python train_demo.py \
  --data-dir Data/1.DLPFC/151673 \
  --metadata Data/1.DLPFC/151673/metadata.tsv \
  --id-column Barcode \
  --label-column layer_guess \
  --device cuda:0 \
  --output Results/demo_151673
```

Omit `--metadata` to train without annotation-based evaluation. Annotations are used only to calculate ARI/NMI, not to train the model. Use `--device cpu` for CPU execution. For another section, supply its data directory and the appropriate `--n-clusters`; the example defaults to seven clusters.

The output directory must be new or empty.

### Outputs

```text
Results/demo_151673/
├── Ours_predictions.csv
├── Ours_embedding.tsv
└── metrics.csv
```

- `Ours_predictions.csv`: spot barcodes, spatial coordinates, optional annotations, and predicted domains.
- `Ours_embedding.tsv`: the reconstructed representation used for clustering, indexed by spot barcode. Its width corresponds to the reconstructed expression features, rather than necessarily the 64-dimensional encoder latent space.
- `metrics.csv`: spot counts, inferred domain count, total run time, and ARI/NMI when valid annotations are supplied.

## HBC analysis from saved predictions

`analysis/plot_hbc.py` analyzes a saved HBC clustering result without retraining TRACE. The current source package does not include the HBC prediction file or raw expression data; provide these inputs explicitly.

### Inputs

1. An HBC prediction CSV containing unique `spot_id` values and these columns:

   ```text
   spot_id,spatial_x,spatial_y,ground_truth,Ours
   ```

2. The matching raw `filtered_feature_bc_matrix.h5` expression matrix.

The prediction CSV must contain usable regional annotations in `ground_truth` for annotation-guided marker displays. Expression and predictions are aligned by barcode. Domain IDs must correspond to the selected saved prediction file.

The original HBC data are the 10x Genomics **Human Breast Cancer (Block A, Section 1)** dataset. Region annotations and saved predictions are separate analysis inputs and must match that section.

### Run

```bash
python analysis/plot_hbc.py \
  --predictions /path/to/HBC/Ours_predictions.csv \
  --counts /path/to/HBC/filtered_feature_bc_matrix.h5 \
  --out-dir figures/HBC \
  --domain-a 2 \
  --domain-b 5 \
  --genes TFF1 ABCC11 CPB1 CRISP3
```

For the original selected result, domains 2 and 5 were matched to DCIS/LCIS-associated and IDC-associated regions. Check the generated domain-to-annotation matching table before interpreting another prediction file. Cluster numbering can change after retraining.

Always provide `--predictions`, `--counts`, and `--out-dir`; the current script retains historical defaults that should not be used as repository-relative data paths.

Spatial gene maps show measured, library-normalized expression from the input count matrix. Domain matching assigns display labels and colors to the predicted clusters. Pairwise differential expression compares spots assigned to the selected predicted domains.

Figures are saved as PNG, SVG and PDF, together with analysis tables. GO enrichment requires `--gmt`; an independent ten-domain comparison requires `--predictions-k10`. Those optional panels are skipped when their inputs are not supplied.

For gene-name resolution, optional `--gene-gtf` or `--gene-name-map` inputs provide an explicit annotation source. Without these inputs, the script retains the feature names stored in the count matrix.
