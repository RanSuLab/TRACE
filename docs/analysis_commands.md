# Analysis commands

Replace result names with the directories you selected. These entry points use explicit inputs.

```bash
python analysis/plot_hbc.py \
  --predictions examples/saved_results/HBC/Ours_predictions.csv \
  --counts Data/3.Human_Breast_Cancer/filtered_feature_bc_matrix.h5 \
  --out-dir figures/HBC --domain-a 2 --domain-b 5 \
  --genes TFF1 ABCC11 CPB1 CRISP3
```
