#!/usr/bin/env python3
"""Biological validation of SAVED Ours Human Breast Cancer domains.

No model is trained or benchmark method loaded. Input domains and coordinates
come from Ours_predictions.csv; expression comes from the original 10x counts.
The panels follow the questions in DeepST Figure 4, without claiming that
the underlying annotations or domain numbers are identical.

Example (run from GraphST-main; defaults resolve next to this script):
  python plot_ours_hbc_validation.py

The default Figure F contrast uses saved domain 2 (mostly DCIS/LCIS_4)
versus domain 5 (mostly IDC_4), with domain 4 (mostly Healthy_1) as a
third descriptive group. It writes three standalone panels, a combined
three-panel PNG/SVG/PDF, and selected-gene pairwise statistics. The
histology names come from an overlap with saved annotations; they are
not model predictions. Spot-level P values are exploratory because nearby
spots are spatially correlated.

Override inputs or restrict domains when needed:
  python plot_ours_hbc_validation.py \
    --predictions constra/DeepST-main/Results/Ours_others_results/Human_Breast_Cancer/radius_50/Ours_predictions.csv \
    --counts Data/3.Human_Breast_Cancer/filtered_feature_bc_matrix.h5 \
    --out-dir figures/Ours_Human_Breast_Cancer_validation \
    --domain-a 2 --domain-b 5 --comparison-domains 2 5 4

Optional: --predictions-k10 path/to/independently_saved_10_domain_predictions.csv
          --gmt path/to/GO_Biological_Process.gmt
          --histology path/to/tissue_hires_image.png

The Embryo-style comparison matches each saved Ours domain to a pathology
label by one-to-one overlap (Hungarian assignment), then selects a marker for
THAT LABEL, using labeled spots versus all other labeled spots. Its upper row
shows only predicted membership; its lower row shows expression in ALL spots.
Label-guided gene selection on this sample is descriptive, not independent
validation of the predictions.

All measured genes, including lncRNAs, are eligible for automatic figures.
Display names are resolved by H5 feature ID, optionally using --gene-gtf
and --gene-name-map. Unknown genes keep their exact original identifiers.
No replacement or aggregation changes the expression measurements.

Requires numpy, pandas, scipy, matplotlib, h5py. Outputs PNG/SVG/PDF per
figure, CSV for numerical results, and analysis_status.json with provenance.
Within predicted_domain_label_markers/, domain/ and expression/ contain
separate titled panels with identical spatial geometry; the paired and
paginated composites remain available.
Their spot area adapts to displayed interspot spacing, not the clustering
refinement radius (e.g. radius=50) used to generate the saved labels.
"""
import sys
from pathlib import Path as _ReleasePath
sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))

import argparse
import gzip
import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

LOG = logging.getLogger("hbc_validation")
COLORS = ("#4E79A7", "#E15759", "#76B7B2", "#F28E2B", "#59A14F",
          "#B07AA1", "#EDC948", "#9C755F", "#FF9DA7", "#BAB0AC",
          "#2F6B8A", "#BD4D53", "#448D85", "#CE6B1A", "#43804A",
          "#885882", "#B49A28", "#745747", "#D6798A", "#6B747D")


def save(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=dpi if ext == "png" else None,
                    bbox_inches="tight", pad_inches=.04, facecolor="white")
    plt.close(fig)
    LOG.info("Saved %s.{png,svg,pdf}", path)


def key(value):
    return str(value).strip().casefold()


def sort_labels(values):
    return sorted(set(str(v) for v in values),
                  key=lambda v: (0, int(v)) if v.isdigit() else (1, v))


def read_predictions(path, column="Ours"):
    frame = pd.read_csv(path, dtype={"spot_id": str}, low_memory=False)
    required = {"spot_id", "spatial_x", "spatial_y", column}
    if not required.issubset(frame):
        raise ValueError("{} needs columns {}".format(path, sorted(required)))
    if frame.spot_id.isna().any() or frame.spot_id.duplicated().any():
        raise ValueError("Duplicate or missing prediction spot IDs: {}".format(path))
    if frame[column].isna().any():
        raise ValueError("Missing predicted domains: {}".format(path))
    frame[column] = frame[column].astype(str)
    for col in ("spatial_x", "spatial_y"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[["spatial_x", "spatial_y"]].to_numpy()).all():
        raise ValueError("Non-finite spatial coordinates")
    return frame.set_index("spot_id", drop=False)


def read_10x(path, spot_ids):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install h5py in the analysis environment") from exc
    with h5py.File(path, "r") as handle:
        group = handle["matrix"] if "matrix" in handle else next(iter(handle.values()))
        n_genes, n_spots = map(int, group["shape"][:])
        matrix = sparse.csc_matrix((group["data"][:], group["indices"][:],
                                    group["indptr"][:]), shape=(n_genes, n_spots)).T.tocsr()
        ids = [v.decode() if isinstance(v, bytes) else str(v) for v in group["barcodes"][:]]
        features = group["features"] if "features" in group else group
        names_key = "name" if "name" in features else "gene_names"
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in features[names_key][:]]
        gene_ids = [v.decode() if isinstance(v, bytes) else str(v)
                    for v in features["id"][:]] if "id" in features else names[:]
    if len(ids) != len(set(ids)) or len(spot_ids) != len(set(spot_ids)):
        raise ValueError("Nonunique spot IDs")
    missing = set(spot_ids) - set(ids)
    if missing:
        raise ValueError("{} prediction IDs missing in 10x counts; examples {}"
                         .format(len(missing), sorted(missing)[:5]))
    order = pd.Index(ids).get_indexer(spot_ids)
    matrix = matrix[order]
    if matrix.shape[1] != len(names) or len(gene_ids) != len(names):
        raise ValueError("10x feature dimension mismatch")
    # Sum columns with repeated symbols so a marker cannot be selected at random.
    normalized_names = [key(v) for v in names]
    unique = list(dict.fromkeys(normalized_names))
    if len(unique) < len(names):
        indices = pd.Index(unique).get_indexer(normalized_names)
        mapping = sparse.csr_matrix((np.ones(len(names)),
                                     (np.arange(len(names)), indices)),
                                    shape=(len(names), len(unique)))
        matrix = (matrix @ mapping).tocsr()
        grouped_ids = [[] for _ in unique]
        for idx, gene_id in zip(indices, gene_ids):
            grouped_ids[idx].append(gene_id)
        gene_ids = [tuple(group) for group in grouped_ids]
        names = [names[normalized_names.index(v)] for v in unique]
        LOG.info("Collapsed %d duplicate feature symbols", len(indices) - len(unique))
    else:
        gene_ids = [(gene_id,) for gene_id in gene_ids]
    if matrix.nnz and (np.min(matrix.data) < 0 or not np.isfinite(matrix.data).all()):
        raise ValueError("Counts must be finite and nonnegative")
    return matrix.astype(np.float32), names, gene_ids


def resolve_gene_labels(genes, gene_ids, gtf=None, name_map=None):
    """Resolve figure labels by stable IDs; keep expression indexed by H5 names."""
    # These are source-specific, verified names, not inferred protein aliases.
    known = {
        "ENSG00000236272": ("SCUBE1-AS2", "HGNC:40633; lncRNA"),
        "ENSG00000254695": ("lnc-INSC-2", "LNCipedia v5.2 locus ID; lncRNA; not an HGNC symbol"),
    }
    annotation = {}
    if gtf:
        opener = gzip.open if str(gtf).endswith(".gz") else open
        with opener(gtf, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 9 or fields[2] != "gene":
                    continue
                attrs = dict(re.findall(r'(\w+) "([^"]+)"', fields[8]))
                if attrs.get("gene_id") and attrs.get("gene_name"):
                    stable = attrs["gene_id"].split(".")[0]
                    value = (attrs["gene_name"],
                             "GTF {} ({})".format(gtf.name,
                                attrs.get("gene_type", attrs.get("gene_biotype", "untyped"))))
                    if stable in annotation and annotation[stable][0] != value[0]:
                        raise ValueError("Conflicting GTF gene names for " + stable)
                    annotation[stable] = value
    if name_map:
        custom = pd.read_csv(name_map, sep="\t" if name_map.suffix.lower() == ".tsv" else ",",
                             dtype=str).fillna("")
        if not {"gene_id", "display_name"}.issubset(custom):
            raise ValueError("--gene-name-map needs gene_id and display_name columns")
        if custom.gene_id.duplicated().any() or (custom.display_name.str.strip() == "").any():
            raise ValueError("--gene-name-map contains duplicate IDs or empty names")
        for row in custom.itertuples(index=False):
            stable = row.gene_id.split(".")[0]
            annotation[stable] = (row.display_name,
                                  "user name map: {}".format(name_map.name))
    records = []
    for gene, ids in zip(genes, gene_ids):
        stables = list(dict.fromkeys(gid.split(".")[0] for gid in ids))
        resolved = []
        for stable in stables:
            candidate = annotation.get(stable)
            if candidate is None or candidate[0] == gene:
                candidate = known.get(stable, candidate)
            resolved.append(candidate)
        if len(stables) == 1 and resolved[0]:
            label, source = resolved[0]
        else:
            label, source = gene, "original H5 name; unresolved or multiple feature IDs"
        records.append({"gene": gene, "feature_ids": ";".join(ids),
                        "display_name": label, "name_source": source})
    return pd.DataFrame.from_records(records)


def normalized_counts(counts):
    totals = np.asarray(counts.sum(axis=1)).ravel()
    norm = counts.multiply((1e4 / np.maximum(totals, 1))[:, None]).tocsr()
    return norm


def highly_variable_genes(norm, counts, genes, n_top, out_dir):
    """Select genes by variance of log1p(CPM10k); no annotations are used."""
    logged = norm.copy().tocsr()
    np.log1p(logged.data, out=logged.data)
    mean = np.asarray(logged.mean(axis=0)).ravel()
    second = np.asarray(logged.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - mean * mean, 0)
    prevalence = np.asarray(counts.getnnz(axis=0)).ravel() / counts.shape[0]
    candidates = np.flatnonzero((prevalence >= .01) & (variance > 0))
    ranked = candidates[np.argsort(-variance[candidates], kind="stable")]
    chosen = ranked[:min(n_top, len(ranked))]
    if len(chosen) < 2:
        raise ValueError("Fewer than two nonconstant genes meet 1% prevalence")
    table = pd.DataFrame({"gene": genes, "log1p_mean": mean,
                          "log1p_variance": variance, "spot_prevalence": prevalence,
                          "selected_hvg": np.isin(np.arange(len(genes)), chosen)})
    table.sort_values("log1p_variance", ascending=False).to_csv(
        out_dir / "HVG_gene_statistics.csv", index=False)
    return chosen


def fdr_bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    p = np.where(np.isfinite(p), np.clip(p, 0, 1), 1)
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * len(p) /
                                    np.arange(1, len(p) + 1))[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1)
    return output


def de_test(norm, genes, group_a, group_b, block_size=256):
    """Welch t-test on log1p normalized counts; log2FC on normalized means."""
    if int(group_a.sum()) < 3 or int(group_b.sum()) < 3:
        raise ValueError("DE requires at least three spots in each group")
    result = []
    for start in range(0, norm.shape[1], block_size):
        stop = min(start + block_size, norm.shape[1])
        a = norm[group_a, start:stop].toarray()
        b = norm[group_b, start:stop].toarray()
        raw_a, raw_b = np.mean(a, axis=0), np.mean(b, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = stats.ttest_ind(np.log1p(a), np.log1p(b), axis=0,
                                equal_var=False).pvalue
        logfc = np.log2((raw_a + .01) / (raw_b + .01))
        prevalence = np.maximum(np.mean(a > 0, axis=0), np.mean(b > 0, axis=0))
        for i, gene in enumerate(genes[start:stop]):
            result.append((gene, float(logfc[i]), float(p[i]) if np.isfinite(p[i]) else 1.,
                           float(prevalence[i]), float(raw_a[i]), float(raw_b[i])))
    table = pd.DataFrame(result, columns=["gene", "log2FC", "pvalue", "prevalence",
                                          "mean_a_cpm1e4", "mean_b_cpm1e4"])
    tested = (table.prevalence >= .01) & ((table.mean_a_cpm1e4 + table.mean_b_cpm1e4) > .01)
    table.loc[~tested, "pvalue"] = 1.
    table["fdr"] = fdr_bh(table.pvalue)
    return table.sort_values(["fdr", "pvalue", "gene"]).reset_index(drop=True)


def scatter_spatial(ax, xy, labels, palette, point_size=4., legend=False):
    for label in sort_labels(labels):
        mask = labels == label
        ax.scatter(xy[mask, 0], xy[mask, 1], s=point_size, c=palette[label],
                   marker="o", linewidths=0, rasterized=True, label=label)
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.axis("off")
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False,
                  markerscale=2.2, fontsize=7, ncol=1)


def domain_map(frame, column, path, dpi, point_size=4., legend=True):
    labels = frame[column].astype(str).to_numpy()
    ordered = sort_labels(labels)
    palette = {v: COLORS[i % len(COLORS)] for i, v in enumerate(ordered)}
    fig, ax = plt.subplots(figsize=(5., 4.4))
    scatter_spatial(ax, frame[["spatial_x", "spatial_y"]].to_numpy(float),
                    labels, palette, point_size, legend)
    save(fig, path, dpi)
    return palette


def marker_values(norm, genes, symbol):
    lookup = {key(v): i for i, v in enumerate(genes)}
    if key(symbol) not in lookup:
        return None
    return np.log1p(norm[:, lookup[key(symbol)]].toarray().ravel())


def marker_map(xy, values, name, path, dpi, size):
    upper = np.quantile(values, .992)
    if upper <= 0:
        LOG.warning("Marker %s has zero p99.2 abundance; skip spatial map", name)
        return False
    fig, ax = plt.subplots(figsize=(5., 4.4))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=size, linewidths=0,
                    cmap="magma", norm=Normalize(vmin=0, vmax=upper), rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(name, fontsize=10, pad=4)
    fig.colorbar(sc, ax=ax, fraction=.035, pad=.01, label="log(1 + counts per 10k)")
    save(fig, path, dpi)
    return True


def violin(norm, genes, labels, requested, path, dpi, display=None):
    available = [(name, marker_values(norm, genes, name)) for name in requested]
    available = [(name, v) for name, v in available if v is not None]
    if not available:
        return False
    groups = sort_labels(labels)
    fig, axes = plt.subplots(len(available), 1, figsize=(max(6., len(groups) * .37),
                                                  2.0 * len(available)), squeeze=False)
    for ax, (name, values) in zip(axes[:, 0], available):
        data = [values[labels == group] for group in groups]
        parts = ax.violinplot(data, positions=np.arange(len(groups)), showmedians=True,
                              showextrema=False, widths=.8)
        for body in parts["bodies"]:
            body.set_facecolor("#609EA0")
            body.set_edgecolor("none")
            body.set_alpha(.85)
        parts["cmedians"].set_color("#30363D")
        ax.set_ylabel((display or {}).get(key(name), name), rotation=0, ha="right", va="center")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=.4, color="#DDDDDD")
    for ax in axes[:, 0]:
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=90, fontsize=7)
    axes[-1, 0].set_xlabel("Saved Ours domain")
    fig.tight_layout()
    save(fig, path, dpi)
    return True


def correlation(norm, labels, path, out_dir, dpi):
    groups = sort_labels(labels)
    means = np.vstack([np.asarray(norm[labels == group].mean(axis=0)).ravel()
                       for group in groups])
    keep = means.sum(axis=0) > 0
    data = np.log1p(means[:, keep])
    corr = np.corrcoef(data)
    pd.DataFrame(corr, index=groups, columns=groups).to_csv(out_dir / (path.stem + ".csv"))
    fig, ax = plt.subplots(figsize=(max(5, len(groups) * .31), max(4.7, len(groups) * .30)))
    off_diag = corr[~np.eye(len(groups), dtype=bool)]
    lower = float(np.nanmin(off_diag)) if off_diag.size else -1.
    # Use the data range, and disclose both its range and the gene subset in
    # the accompanying CSV/status. A fixed [-1,1] range hides real differences.
    lower = max(-1., np.floor(lower * 100) / 100)
    upper = 1. if lower < 1 else 1.00001
    im = ax.imshow(corr, cmap="viridis", vmin=lower, vmax=upper,
                   interpolation="nearest")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=90, fontsize=7)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=7)
    ax.set_xlabel("Ours domain")
    ax.set_ylabel("Ours domain")
    fig.colorbar(im, ax=ax, label="Pearson r", fraction=.046, pad=.02)
    save(fig, path, dpi)


def label_marker_choice(table, norm, inside, outside, gene_lookup):
    """Choose one enriched HVG in the annotated label, never in predicted domains."""
    candidates = table[(table.fdr < .05) & (table.log2FC >= 1.) &
                       (table.prevalence >= .10)].copy()
    if candidates.empty:
        return None
    indexes = [gene_lookup[key(g)] for g in candidates.gene]
    presence_in = np.asarray((norm[inside][:, indexes] > 0).mean(axis=0)).ravel()
    presence_out = np.asarray((norm[outside][:, indexes] > 0).mean(axis=0)).ravel()
    candidates["presence_in"] = presence_in
    candidates["presence_out"] = presence_out
    candidates = candidates[(candidates.presence_in >= .10) &
                            (candidates.presence_in >= 1.3 * candidates.presence_out)]
    if candidates.empty:
        return None
    return candidates.sort_values(["log2FC", "fdr", "gene"],
                                  ascending=[False, True, True]).iloc[0]


def match_predicted_to_annotations(predicted, truth, valid):
    """One-to-one overlap matching; unmatched/weak matches remain explicit."""
    domains = sort_labels(predicted)
    classes = sort_labels(truth[valid])
    if not classes:
        raise ValueError("No labeled spots for label-guided marker analysis")
    overlap = np.asarray([[int(np.sum((predicted == domain) &
                                      (truth == label) & valid))
                           for label in classes] for domain in domains])
    rows, cols = linear_sum_assignment(-overlap)
    assignments = {domains[i]: (classes[j], int(overlap[i, j]))
                   for i, j in zip(rows, cols) if overlap[i, j] > 0}
    result = []
    for domain in domains:
        label, n = assignments.get(domain, (None, 0))
        size = int(np.sum(predicted == domain))
        result.append({"predicted_domain": domain, "matched_annotation": label or "",
                       "overlap_spots": n, "domain_spots": size,
                       "match_fraction": n / size,
                       "annotation_spots": int(np.sum((truth == label) & valid)) if label else 0,
                       "match_method": "one-to-one maximum overlap; display only"})
    return result


def paired_spot_area(ax, xy, fill=1.10):
    """Scatter area in pt^2 derived from median on-screen spot spacing."""
    ax.figure.canvas.draw()
    screen = ax.transData.transform(xy)
    distance = cKDTree(screen).query(screen, k=2)[0][:, 1]
    usable = distance[np.isfinite(distance) & (distance > 0)]
    if not len(usable):
        raise ValueError("Cannot infer spot spacing from duplicate spatial coordinates")
    spacing_pt = np.median(usable) * 72. / ax.figure.dpi
    return float((spacing_pt * fill) ** 2)


def clean_spatial_axes(ax, xy):
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    dx, dy = max(x_max-x_min, 1), max(y_max-y_min, 1)
    ax.set_xlim(x_min-.015*dx, x_max+.015*dx)
    ax.set_ylim(y_min-.015*dy, y_max+.015*dy)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def predicted_marker_axes(ax_top, ax_bottom, xy, labels, domain, annotation,
                          gene, values, color, upper, fill, titles=True):
    """Matched geometry, orientation and spot appearance in both rows."""
    selected = labels == domain
    for ax in (ax_top, ax_bottom):
        clean_spatial_axes(ax, xy)
    top_area = paired_spot_area(ax_top, xy, fill)
    bottom_area = paired_spot_area(ax_bottom, xy, fill)
    ax_top.scatter(xy[:, 0], xy[:, 1], s=top_area, color="#DCDDDD",
                   linewidths=0, rasterized=True)
    ax_top.scatter(xy[selected, 0], xy[selected, 1], s=top_area,
                   color=color, linewidths=0, rasterized=True)
    ax_bottom.scatter(xy[:, 0], xy[:, 1], c=values, s=bottom_area,
                      cmap="viridis", vmin=0, vmax=upper, linewidths=0,
                      rasterized=True)
    if titles:
        ax_top.set_title(annotation, fontsize=9, pad=3)
        ax_bottom.set_title(gene, fontsize=9, pad=3)


def embryo_style_markers(norm, genes, predicted, annotation, valid, xy, hvgs,
                         focus, out_dir, dpi, spot_fill, min_match_fraction, display):
    """Pair SAVED predicted domains with markers selected from matched labels."""
    lookup = {key(g): i for i, g in enumerate(genes)}
    hvg_genes = [genes[i] for i in hvgs]
    mapping = match_predicted_to_annotations(predicted, annotation, valid)
    pd.DataFrame(mapping).to_csv(out_dir / "predicted_to_annotation_mapping.csv", index=False)
    selections = []
    figures = []
    tested_labels = {}
    for domain in focus:
        record = next(row for row in mapping if row["predicted_domain"] == domain)
        label = record["matched_annotation"]
        if not label or record["match_fraction"] < min_match_fraction:
            LOG.warning("Skip domain %s: matched annotation %s, overlap %.1f%% < %.1f%%",
                        domain, label or "none", 100*record["match_fraction"],
                        100*min_match_fraction)
            continue
        count = record["domain_spots"]
        inside = valid & (annotation == label)
        outside = valid & (annotation != label)
        if inside.sum() < 3 or outside.sum() < 3:
            LOG.warning("Skip %s: insufficient labeled spots for label vs rest", label)
            continue
        if label not in tested_labels:
            table = de_test(norm[:, hvgs], hvg_genes, inside, outside)
            table.insert(1, "display_name", table.gene.map(
                lambda name: display.get(key(name), name)))
            table.to_csv(out_dir / ("annotation_{}_HVG_vs_rest.csv".format(
                re.sub(r"[^A-Za-z0-9_-]", "_", label))), index=False)
            tested_labels[label] = label_marker_choice(table, norm, inside, outside, lookup)
        chosen = tested_labels[label]
        if chosen is None:
            LOG.warning("Skip %s: no annotation-enriched HVG meets marker thresholds", label)
            continue
        gene = chosen.gene
        gene_title = display.get(key(gene), gene)
        values = marker_values(norm, genes, gene)
        upper = float(np.quantile(values, .992))
        if upper <= 0:
            continue
        predicted_mask = predicted == domain
        predicted_values = np.expm1(values)
        pred_fc = np.log2((predicted_values[predicted_mask].mean()+.01) /
                          (predicted_values[~predicted_mask].mean()+.01))
        selections.append({"predicted_domain": domain, "n_predicted_spots": count,
                           "matched_annotation": label,
                           "match_fraction": record["match_fraction"],
                           "marker_gene": gene, "marker_display_name": gene_title,
                           "marker_selected_from": "annotation vs other labeled spots",
                           "label_log2FC_vs_rest": float(chosen.log2FC),
                           "label_fdr": float(chosen.fdr),
                           "label_presence_in": float(chosen.presence_in),
                           "label_presence_out": float(chosen.presence_out),
                           "predicted_domain_marker_log2FC_vs_rest": float(pred_fc),
                           "marker_display_p99_2": upper})
        figures.append((domain, label, gene_title, values, upper))
        fig, axes = plt.subplots(2, 1, figsize=(3.2, 6.3))
        predicted_marker_axes(axes[0], axes[1], xy, predicted, domain, label,
                              gene_title, values,
                              COLORS[sort_labels(predicted).index(domain) % len(COLORS)],
                              upper, spot_fill)
        fig.subplots_adjust(hspace=.05)
        safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label)
        save(fig, out_dir / ("{}_marker_{}".format(
            safe_label, re.sub(r"[^A-Za-z0-9_-]", "_", gene_title))), dpi)
        # Separate panels share the same axes rectangle; figure-level titles
        # do not displace either spatial coordinate system.
        stem = safe_label
        for source, suffix in (("domain", "predicted"), ("expression", "expression_" +
                                re.sub(r"[^A-Za-z0-9_-]", "_", gene_title))):
            separate = plt.figure(figsize=(3.2, 3.35))
            single = separate.add_axes([.01, .01, .98, .90])
            separate.text(.5, .975,
                          label if source == "domain" else gene_title,
                          ha="center", va="top", fontsize=10, color="#202020")
            clean_spatial_axes(single, xy)
            area = paired_spot_area(single, xy, spot_fill)
            if source == "domain":
                single.scatter(xy[:, 0], xy[:, 1], s=area, color="#DCDDDD",
                               linewidths=0, rasterized=True)
                mask = predicted == domain
                single.scatter(xy[mask, 0], xy[mask, 1], s=area,
                               color=COLORS[sort_labels(predicted).index(domain) % len(COLORS)],
                               linewidths=0, rasterized=True)
            else:
                single.scatter(xy[:, 0], xy[:, 1], c=values, s=area,
                               cmap="viridis", vmin=0, vmax=upper,
                               linewidths=0, rasterized=True)
            save(separate, out_dir / source / (stem + "_" + suffix), dpi)
    pd.DataFrame(selections, columns=["predicted_domain", "n_predicted_spots",
                     "matched_annotation", "match_fraction", "marker_gene", "marker_display_name",
                     "marker_selected_from", "label_log2FC_vs_rest", "label_fdr",
                     "label_presence_in", "label_presence_out",
                     "predicted_domain_marker_log2FC_vs_rest",
                     "marker_display_p99_2"]).to_csv(out_dir / "domain_marker_selections.csv", index=False)
    for start in range(0, len(figures), 6):
        subset = figures[start:start+6]
        fig, axes = plt.subplots(2, len(subset), figsize=(2.8*len(subset), 6.2),
                                 squeeze=False)
        for col, (domain, label, gene_title, values, upper) in enumerate(subset):
            predicted_marker_axes(axes[0, col], axes[1, col], xy, predicted,
                                  domain, label, gene_title, values,
                                  COLORS[sort_labels(predicted).index(domain) % len(COLORS)],
                                  upper, spot_fill)
        fig.subplots_adjust(wspace=.02, hspace=.04)
        save(fig, out_dir / ("predicted_domains_HVG_markers_page_{}".format(
            start//6 + 1)), dpi)
    return len(figures)


def volcano(table, a, b, path, dpi, y_max=None, label_genes=(), display=None):
    """Display a robust FDR range; retain every gene in the exported DE table."""
    fig, ax = plt.subplots(figsize=(5.1, 4.5))
    result = draw_volcano(ax, table, a, b, y_max, label_genes, display)
    save(fig, path, dpi)
    return result


def draw_volcano(ax, table, a, b, y_max=None, label_genes=(), display=None):
    """Draw the same volcano into an individual or composite panel."""
    x = table.log2FC.to_numpy()
    fdr = table.fdr.to_numpy()
    positive = fdr[(fdr > 0) & np.isfinite(fdr)]
    positive_y = -np.log10(np.clip(positive, 1e-300, 1))
    if y_max is None:
        # Ignore the extreme tail for axis selection; this only changes display.
        y_max = float(np.clip(np.ceil(np.percentile(positive_y, 99.5) / 5) * 5,
                              15, 45)) if len(positive_y) else 25.
    y = -np.log10(np.maximum(fdr, 1e-300))
    visible = (fdr > 0) & np.isfinite(fdr) & np.isfinite(x) & (y <= y_max)
    significant = (table.fdr.to_numpy() < .05) & (np.abs(x) >= 1.)
    ax.scatter(x[~significant & visible], y[~significant & visible],
               s=5, color="#C5CBD0", alpha=.58,
               linewidths=0, rasterized=True)
    for mask, color in ((significant & (x > 0), "#CB5959"),
                        (significant & (x < 0), "#4E79A7")):
        shown = mask & visible
        ax.scatter(x[shown], y[shown], s=8, color=color,
                   alpha=.7, linewidths=0, rasterized=True)
    chosen = list(dict.fromkeys(str(g) for g in label_genes))
    for i, gene in enumerate(chosen):
        row = table.loc[table.gene.str.casefold() == gene.casefold()]
        if row.empty:
            LOG.warning("Volcano label %s absent from tested genes", gene)
            continue
        index = int(row.index[0])
        if not visible[index]:
            LOG.warning("Volcano label %s lies outside the plotted FDR range", gene)
            continue
        ax.annotate((display or {}).get(key(gene), gene), (x[index], y[index]), fontsize=8,
                    xytext=((10 if x[index] >= 0 else -10),
                            12 + (i % 3) * 10), textcoords="offset points",
                    ha="left" if x[index] >= 0 else "right",
                    arrowprops={"arrowstyle": "-", "lw": .5, "color": "#555555"})
    ax.axhline(-np.log10(.05), lw=.7, color="#888888", ls="--")
    ax.axvline(-1, lw=.7, color="#888888", ls="--")
    ax.axvline(1, lw=.7, color="#888888", ls="--")
    ax.set_ylim(0, y_max)
    ax.set_xlabel("log2 FC (domain {} / {})".format(a, b))
    ax.set_ylabel("−log10 adjusted P")
    ax.spines[["top", "right"]].set_visible(False)
    omitted_zero = int((fdr == 0).sum())
    omitted_high = int(((fdr > 0) & (y > y_max)).sum())
    LOG.info("Volcano display y=0..%g; omitted %d zero-FDR and %d above-range genes; "
             "all genes retained in DE CSV", y_max, omitted_zero, omitted_high)
    return {"display_y_max": y_max, "omitted_zero_fdr": omitted_zero,
            "omitted_above_display_max": omitted_high,
            "display_only_filter": "all genes retained in DE CSV",
            "significant_fdr_0.05_abs_log2fc_1": int(significant.sum())}


def comparison_gene_names(table, requested):
    """Pick one enriched gene per direction from the actual two-domain DE test."""
    if requested:
        return list(dict.fromkeys(requested)), "user supplied"
    candidate = table[(table.fdr < .05) & (table.log2FC.abs() >= 1) &
                      (table.prevalence >= .10)].copy()
    candidate = candidate[~candidate.gene.str.upper().str.startswith(
        ("MT-", "RPL", "RPS"))]
    positive = candidate[candidate.log2FC > 0].sort_values(
        ["fdr", "log2FC"], ascending=[True, False])
    negative = candidate[candidate.log2FC < 0].sort_values(
        ["fdr", "log2FC"], ascending=[True, True])
    chosen = [part.iloc[0].gene for part in (positive, negative) if len(part)]
    if len(chosen) == 1:
        remaining = candidate[~candidate.gene.isin(chosen)].sort_values(
            ["fdr", "gene"])
        if len(remaining):
            chosen.append(remaining.iloc[0].gene)
    return chosen, "FDR < .05, absolute log2FC >= 1, prevalence >= .10; prefer one gene per direction, then next-best if necessary"


def draw_comparison_spatial(ax, xy, labels, domains, colors):
    for group in domains:
        mask = labels == group
        ax.scatter(xy[mask, 0], xy[mask, 1], s=7, c=colors[group],
                   linewidths=0, rasterized=True)
        center = np.median(xy[mask], axis=0)
        ax.text(center[0], center[1], group, fontsize=10, weight="bold",
                ha="center", va="center", color="#24292E",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8,
                      "boxstyle": "round,pad=0.15"})
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].max(), xy[:, 1].min())
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def format_fdr(value):
    return "<1e-300" if value <= 0 else "{:.2g}".format(value)


def draw_comparison_violin(axes, norm, genes, labels, domains, gene_names,
                           colors, pairwise, display):
    for ax, gene in zip(axes, gene_names):
        values = marker_values(norm, genes, gene)
        series = [values[labels == group] for group in domains]
        parts = ax.violinplot(series, positions=np.arange(3), widths=.7,
                              showmeans=False, showmedians=False, showextrema=False)
        for body, group in zip(parts["bodies"], domains):
            body.set_facecolor(colors[group])
            body.set_edgecolor("none")
            body.set_alpha(.87)
        for j, sample in enumerate(series):
            lo, med, hi = np.quantile(sample, [.25, .5, .75])
            ax.plot([j, j], [lo, hi], color="#26313A", lw=1.3, zorder=4)
            ax.scatter(j, med, s=14, color="white", edgecolor="#26313A",
                       linewidth=.55, zorder=5)
        ceiling = max(float(max(np.max(sample) for sample in series)), .25)
        step = max(ceiling * .095, .12)
        for level, (i, j) in enumerate(((0, 1), (1, 2), (0, 2))):
            pair = pairwise[(gene, domains[i], domains[j])]
            y = ceiling + step * (level + .38)
            ax.plot([i, i, j, j], [y-step*.12, y, y, y-step*.12],
                    color="#4A5158", lw=.65)
            ax.text((i+j)/2, y+step*.05, "q="+format_fdr(pair["fdr"]),
                    ha="center", va="bottom", fontsize=7, color="#343B42")
        ax.set_ylim(0, ceiling + step*3.8)
        ax.set_xlim(-.45, 2.45)
        ax.set_title(display.get(key(gene), gene), loc="left", fontsize=10,
                     weight="bold", pad=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels(domains, fontsize=8)
        ax.set_ylabel("log1p(CPM10k)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
    axes[-1].set_xlabel("Ours domain", fontsize=8)


def comparison_triptych(norm, genes, frame, labels, domains, de_ab, gene_request,
                        y_max, dpi, out_dir, display):
    """Save three matched panels plus a vector composite from saved domains."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen, selection = comparison_gene_names(de_ab, gene_request)
    lookup = {key(g): g for g in genes}
    chosen = [lookup[key(g)] for g in chosen if key(g) in lookup]
    if not chosen:
        return {"skipped": "No requested or bidirectional qualifying genes found"}
    if len(chosen) < 2:
        LOG.warning("Only one eligible comparison gene: %s", chosen)
    chosen = chosen[:2]
    colors = {domains[0]: "#E58B39", domains[1]: "#479B77",
              domains[2]: "#8574A4"}
    xy = frame[["spatial_x", "spatial_y"]].to_numpy(float)
    pair_tables = {(domains[0], domains[1]): de_ab}
    for i, j in ((0, 2), (1, 2)):
        left, right = domains[i], domains[j]
        pair_tables[(left, right)] = de_test(norm, genes, labels == left,
                                             labels == right)
    rows = []
    pairwise = {}
    for gene in chosen:
        for i, j in ((0, 1), (1, 2), (0, 2)):
            left, right = domains[i], domains[j]
            record = pair_tables[(left, right)]
            item = record.loc[record.gene.str.casefold() == gene.casefold()].iloc[0]
            row = {"gene": gene, "domain_a": left, "domain_b": right,
                   "spots_a": int((labels == left).sum()),
                   "spots_b": int((labels == right).sum()),
                   "log2FC": float(item.log2FC), "pvalue": float(item.pvalue),
                   "fdr": float(item.fdr)}
            rows.append(row)
            pairwise[(gene, left, right)] = row
    pairwise_df = pd.DataFrame(rows)
    pairwise_df.insert(1, "display_name",
                       pairwise_df.gene.map(lambda gene: display.get(key(gene), gene)))
    pairwise_df.to_csv(out_dir / "F_three_domain_pairwise_DE.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    draw_comparison_spatial(ax, xy, labels, domains, colors)
    save(fig, out_dir / "F_three_domain_locations", dpi)
    fig, axes = plt.subplots(len(chosen), 1, figsize=(3.8, 3.2*len(chosen)),
                             squeeze=False)
    draw_comparison_violin(axes[:, 0], norm, genes, labels, domains,
                           chosen, colors, pairwise, display)
    fig.tight_layout(h_pad=.45)
    save(fig, out_dir / "F_three_domain_gene_violins", dpi)

    fig = plt.figure(figsize=(14.3, 5.7))
    grid = fig.add_gridspec(2, 3, width_ratios=[3.5, 4.3, 3.4],
                           wspace=.30, hspace=.50)
    spatial_ax = fig.add_subplot(grid[:, 0])
    volcano_ax = fig.add_subplot(grid[:, 1])
    violin_axes = [fig.add_subplot(grid[i, 2]) for i in range(len(chosen))]
    draw_comparison_spatial(spatial_ax, xy, labels, domains, colors)
    draw_volcano(volcano_ax, de_ab, domains[0], domains[1], y_max, display=display)
    draw_comparison_violin(violin_axes, norm, genes, labels, domains,
                           chosen, colors, pairwise, display)
    spatial_ax.set_title("Saved Ours domains", fontsize=10)
    volcano_ax.set_title("Differential expression", fontsize=10)
    save(fig, out_dir / "F_three_domain_comparison", dpi)
    return {"domains": domains, "genes": chosen,
            "display_names": [display.get(key(g), g) for g in chosen],
            "gene_selection": selection,
            "pairwise_test": "Welch t-test on spot-level log1p CPM10k; BH across all genes within each contrast; exploratory",
            "outputs": ["F_three_domain_locations", "F_three_domain_gene_violins",
                        "F_three_domain_comparison", "F_three_domain_pairwise_DE.csv"]}


def deg_heatmap(norm, genes, labels, a, b, table, path, dpi, seed,
                display, cutoff=2., limit=16):
    selected = table[(table.fdr < .05) & (table.log2FC.abs() >= cutoff)]
    top = pd.concat([selected[selected.log2FC > 0].nsmallest(limit, "fdr"),
                     selected[selected.log2FC < 0].nsmallest(limit, "fdr")])
    if top.empty:
        return False
    rng = np.random.default_rng(seed)
    a_idx, b_idx = np.flatnonzero(labels == a), np.flatnonzero(labels == b)
    sampled = np.r_[rng.choice(a_idx, min(160, len(a_idx)), replace=False),
                    rng.choice(b_idx, min(160, len(b_idx)), replace=False)]
    lookup = {key(g): i for i, g in enumerate(genes)}
    indexes = [lookup[key(g)] for g in top.gene]
    matrix = np.log1p(norm[sampled][:, indexes].toarray()).T
    matrix = ((matrix - matrix.mean(axis=1, keepdims=True)) /
              np.maximum(matrix.std(axis=1, keepdims=True), 1e-8))
    matrix = np.clip(matrix, -2.5, 2.5)
    fig, ax = plt.subplots(figsize=(6.5, max(3, .20 * len(top) + 1)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto",
                   interpolation="nearest", rasterized=True)
    ax.axvline(min(160, len(a_idx)) - .5, color="black", lw=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([display.get(key(g), g) for g in top.gene], fontsize=7)
    ax.set_xticks([])
    ax.set_xlabel("Sampled spots: domain {} | domain {}".format(a, b))
    fig.colorbar(im, ax=ax, fraction=.03, pad=.01, label="Gene row z-score")
    save(fig, path, dpi)
    return True


def read_gmt(path):
    terms = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                terms[fields[0]] = {key(v) for v in fields[2:] if v}
    if not terms:
        raise ValueError("No gene sets found in GMT: {}".format(path))
    return terms


def go_enrichment(table, gmt, out_dir, dpi, a, b):
    sets = read_gmt(gmt)
    universe = {key(g) for g in table.loc[table.prevalence >= .01, "gene"]}
    records = []
    for direction, mask in (("up_in_" + a, table.log2FC >= 1),
                            ("up_in_" + b, table.log2FC <= -1)):
        selected = {key(g) for g in table.loc[mask & (table.fdr < .05), "gene"]} & universe
        for term, genes in sets.items():
            hits = selected & genes & universe
            background_size = len(genes & universe)
            if 10 <= background_size <= 500:
                p = stats.hypergeom.sf(len(hits) - 1, len(universe), background_size,
                                       len(selected))
                records.append((direction, term, len(hits), background_size, p,
                                ";".join(sorted(hits))))
    columns = ["direction", "term", "overlap", "term_size", "pvalue", "genes"]
    result = pd.DataFrame(records, columns=columns)
    if result.empty:
        result.to_csv(out_dir / "GO_enrichment.csv", index=False)
        return False
    result["fdr"] = 1.
    for direction in result.direction.unique():
        mask = result.direction == direction
        result.loc[mask, "fdr"] = fdr_bh(result.loc[mask, "pvalue"])
    result.sort_values("fdr").to_csv(out_dir / "GO_enrichment.csv", index=False)
    top = result[(result.fdr < .05) & (result.overlap >= 2)].sort_values("fdr").groupby("direction").head(8)
    if top.empty:
        return False
    fig, ax = plt.subplots(figsize=(6.8, max(3., .33 * len(top) + .8)))
    top = top.sort_values("fdr", ascending=False)
    ax.barh(range(len(top)), -np.log10(np.maximum(top.fdr, 1e-300)),
            color=["#CB5959" if direction == "up_in_" + a else "#4E79A7"
                   for direction in top.direction])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.term, fontsize=8)
    ax.set_xlabel("−log10 FDR")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], color="#CB5959", lw=5, label="Higher in " + a),
                       Line2D([0], [0], color="#4E79A7", lw=5, label="Higher in " + b)],
              frameon=False, fontsize=8)
    save(fig, out_dir / "G_GO_enrichment", dpi)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--predictions", type=Path,
                        default=(project_root / "constra/DeepST-main/Results/"
                                 "Ours_others_results/Human_Breast_Cancer/radius_50/"
                                 "Ours_predictions.csv"),
                        help="Default: canonical saved Ours radius_50 predictions")
    parser.add_argument("--counts", type=Path,
                        default=(project_root / "Data/3.Human_Breast_Cancer/"
                                 "filtered_feature_bc_matrix.h5"),
                        help="Original filtered_feature_bc_matrix.h5, raw counts")
    parser.add_argument("--gene-gtf", type=Path,
                        help="Optional matching GENCODE/Ensembl .gtf[.gz] for ID-based gene names; never filters genes")
    parser.add_argument("--gene-name-map", type=Path,
                        help="Optional curated CSV/TSV with gene_id,display_name columns; overrides GTF names")
    parser.add_argument("--out-dir", type=Path,
                        default=(project_root / "figures/Ours_Human_Breast_Cancer_validation"),
                        help="Default: GraphST-main/figures/Ours_Human_Breast_Cancer_validation")
    parser.add_argument("--predictions-k10", type=Path,
                        help="Independently saved Ours k=10 predictions, exactly same spots")
    parser.add_argument("--gmt", type=Path, help="GO gene sets in GMT format")
    parser.add_argument("--histology", type=Path, help="Optional original H&E image")
    parser.add_argument("--domain-a", default="2",
                        help="Numerator of the volcano contrast; default 2 (DCIS/LCIS_4)")
    parser.add_argument("--domain-b", default="5",
                        help="Denominator of the volcano contrast; default 5 (IDC_4)")
    parser.add_argument("--comparison-domains", nargs=3, default=None,
                        metavar=("FIRST", "SECOND", "THIRD"),
                        help="Three saved Ours domains for the spatial and violin panels; first two must equal --domain-a/--domain-b")
    parser.add_argument("--comparison-genes", nargs="*", default=None,
                        help="Two genes for the three-domain violins; default selects one significant gene enriched in each contrast direction")
    parser.add_argument("--volcano-y-max", type=float, default=None,
                        help="Optional display-only maximum -log10 FDR; larger values are omitted from figure, retained in CSV")
    parser.add_argument("--volcano-label-genes", nargs="*", default=[],
                        help="Only these genes receive volcano labels; default is a clean unlabeled plot")
    parser.add_argument("--genes", nargs="+", default=["TFF1", "ABCC11", "CPB1", "CRISP3"])
    parser.add_argument("--n-hvg", type=int, default=3000,
                        help="Variance-ranked log1p(CPM10k) genes for domain marker analysis")
    parser.add_argument("--focus-domains", nargs="*", default=None,
                        help="Saved Ours domain IDs to include; default: all domains")
    parser.add_argument("--min-match-fraction", type=float, default=.35,
                        help="Minimum matched annotation fraction of a predicted domain (default .35)")
    parser.add_argument("--violin-domains", nargs="*", default=["6", "2", "13", "1", "8"])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--point-size", type=float, default=4.)
    parser.add_argument("--paired-spot-fill", type=float, default=1.10,
                        help="Diameter / median grid spacing for label-marker pairs; default 1.10")
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args(argv)
    if args.comparison_domains is None:
        third = next((v for v in ("4", "2", "5")
                      if v not in (args.domain_a, args.domain_b)), None)
        args.comparison_domains = [args.domain_a, args.domain_b, third]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.n_hvg < 2:
        parser.error("--n-hvg must be at least 2")
    if not 0 <= args.min_match_fraction <= 1:
        parser.error("--min-match-fraction must be between 0 and 1")
    if not .2 <= args.paired_spot_fill <= 1.5:
        parser.error("--paired-spot-fill must be between 0.2 and 1.5")
    if args.volcano_y_max is not None and args.volcano_y_max <= -np.log10(.05):
        parser.error("--volcano-y-max must exceed -log10(0.05)")
    if len(set(args.comparison_domains)) != 3:
        parser.error("--comparison-domains needs three different domain IDs")
    if args.comparison_domains[:2] != [args.domain_a, args.domain_b]:
        parser.error("The first two --comparison-domains must match --domain-a and --domain-b")
    if args.comparison_genes is not None and len(args.comparison_genes) != 2:
        parser.error("--comparison-genes requires exactly two gene names")
    for path in (args.predictions, args.counts):
        if not path.is_file():
            parser.error("Missing input: {}".format(path))
    if args.gene_gtf and not args.gene_gtf.is_file():
        parser.error("Missing gene annotation GTF: {}".format(args.gene_gtf))
    if args.gene_name_map and not args.gene_name_map.is_file():
        parser.error("Missing gene-name map: {}".format(args.gene_name_map))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status = {"inputs": {"predictions": str(args.predictions.resolve()),
                         "counts": str(args.counts.resolve())},
              "methods": {"expression": "raw 10x counts normalized to 10k per spot",
                          "DE": "Welch t-test on log1p normalized counts; BH FDR",
                          "heatmap": "row z-score of log1p normalized counts; sampled spots",
                          "correlation": "Pearson r of log1p domain pseudobulk mean"},
              "created": [], "skipped": {}}

    def made(name):
        status["created"].append(name)

    def skipped(name, why):
        status["skipped"][name] = why
        LOG.warning("Skip %s: %s", name, why)

    try:
        frame = read_predictions(args.predictions)
        xy = frame[["spatial_x", "spatial_y"]].to_numpy(float)
        labels = frame.Ours.to_numpy(str)
        if "ground_truth" in frame and frame.ground_truth.notna().any():
            valid_annotation = (frame.ground_truth.notna().to_numpy() &
                                ~frame.ground_truth.astype(str).str.strip().str.lower()
                                .isin(["", "nan", "none", "unknown", "na", "n/a"]).to_numpy())
            if not valid_annotation.any():
                raise ValueError("No usable annotated spots in saved ground_truth")
            pathology = frame.ground_truth.fillna("Unannotated").astype(str).to_numpy()
            annotation = np.where(valid_annotation, pathology, "")
            domain_map(frame.assign(pathology=pathology), "pathology",
                       args.out_dir / "A_saved_pathology_annotations", args.dpi,
                       args.point_size)
            made("A_saved_pathology_annotations")
            status["inputs"]["pathology"] = "ground_truth in saved Ours_predictions.csv"
            counts_by_label = pd.crosstab(pd.Series(labels, name="predicted_domain"),
                                          pd.Series(pathology, name="saved_annotation"))
            counts_by_label.to_csv(args.out_dir / "predicted_vs_saved_annotation_counts.csv")
        else:
            raise ValueError("Label-guided marker analysis requires ground_truth in saved Ours_predictions.csv")

        domain_map(frame, "Ours", args.out_dir / "B_saved_Ours_domains",
                   args.dpi, args.point_size)
        made("B_saved_Ours_domains")
        skipped("C_domain_number_without_prior", "Saved clustering already specified a domain number; independent model selection was not performed")
        status["n_saved_domains"] = len(set(labels))
        if args.predictions_k10:
            ten = read_predictions(args.predictions_k10)
            if set(ten.index) != set(frame.index):
                raise ValueError("k=10 predictions have different spot IDs")
            ten = ten.loc[frame.index]
            if len(set(ten.Ours)) != 10:
                raise ValueError("--predictions-k10 contains {} domains, expected 10".format(len(set(ten.Ours))))
            if not np.allclose(ten[["spatial_x", "spatial_y"]].to_numpy(float), xy):
                raise ValueError("k=10 saved spatial coordinates disagree with k=20")
            domain_map(ten, "Ours", args.out_dir / "B_independent_k10_domains",
                       args.dpi, args.point_size)
            made("B_independent_k10_domains")
            status["inputs"]["predictions_k10"] = str(args.predictions_k10.resolve())
        else:
            skipped("B_independent_k10_domains", "Supply an actual saved k=10 run")

        counts, genes, feature_ids = read_10x(args.counts, frame.index.tolist())
        name_table = resolve_gene_labels(genes, feature_ids, args.gene_gtf,
                                         args.gene_name_map)
        name_table.to_csv(args.out_dir / "gene_name_resolution.csv", index=False)
        display = {key(row.gene): row.display_name for row in name_table.itertuples()}
        status["gene_naming"] = {
            "rule": "ID-based GTF or curated mapping where available; preserve H5 names otherwise",
            "resolved_renames": int((name_table.gene != name_table.display_name).sum()),
            "n_genes": len(genes), "gene_omissions": 0,
            "mapping_file": "gene_name_resolution.csv"}
        LOG.info("Resolved %d/%d figure gene names by feature ID; no genes excluded",
                 status["gene_naming"]["resolved_renames"], len(genes))
        norm = normalized_counts(counts)
        hvgs = highly_variable_genes(norm, counts, genes, args.n_hvg, args.out_dir)
        status["HVG_method"] = "Top log1p(CPM10k) variance among genes present in >=1% of spots"
        status["n_hvg"] = int(len(hvgs))
        status["n_spots"] = int(len(labels))
        status["n_genes"] = int(len(genes))
        correlation(norm[:, hvgs], labels, args.out_dir / "D_HVG_domain_expression_correlation",
                    args.out_dir, args.dpi)
        made("D_HVG_domain_expression_correlation")
        if args.predictions_k10:
            correlation(norm[:, hvgs], ten.Ours.to_numpy(str), args.out_dir / "D_k10_HVG_domain_expression_correlation",
                        args.out_dir, args.dpi)
            made("D_k10_HVG_domain_expression_correlation")
        focus = sort_labels(labels) if args.focus_domains is None else list(dict.fromkeys(args.focus_domains))
        unknown = set(focus) - set(labels)
        if unknown:
            parser.error("Unknown --focus-domains values: {}".format(sorted(unknown)))
        marker_dir = args.out_dir / "predicted_domain_label_markers"
        marker_dir.mkdir(exist_ok=True)
        n_panels = embryo_style_markers(norm, genes, labels, annotation,
                                       valid_annotation, xy, hvgs, focus,
                                       marker_dir, args.dpi, args.paired_spot_fill,
                                       args.min_match_fraction, display)
        status["label_guided_marker_panel_selection"] = {
            "source": "Saved Ours predictions for upper row; marker DE among annotated labels; raw-count-derived expression for lower row",
            "focus_domains": focus, "panels_saved": n_panels,
            "matching": "Hungarian one-to-one overlap; matching label is for marker selection and display, not predicted spot membership",
            "paired_spot_fill": args.paired_spot_fill,
            "standalone_panels": "domain/ title is the matched annotation; expression/ title is the resolved display name; both share identical spatial geometry",
            "minimum_match_fraction": args.min_match_fraction,
            "thresholds": "Matched annotation vs remaining labeled spots: BH FDR < .05, log2FC >= 1, prevalence >= .10; prevalence in label >= 1.3x outside",
            "inference_limit": "Labels guide both domain matching and marker selection on the same sample; descriptive, not independent validation"}
        if n_panels:
            made("predicted_domain_label_markers")
        else:
            skipped("predicted_domain_label_markers", "No matched annotation passed overlap and marker selection thresholds")
        for gene in args.genes:
            value = marker_values(norm, genes, gene)
            if value is None:
                skipped("expression_" + gene, "Gene absent from 10x matrix")
            elif marker_map(xy, value, display.get(key(gene), gene),
                            args.out_dir / ("E_gene_" + re.sub(r"\W+", "_", display.get(key(gene), gene))),
                            args.dpi, args.point_size):
                made("expression_" + gene)
        if violin(norm, genes, labels, args.genes,
                  args.out_dir / "E_gene_expression_by_domain", args.dpi, display):
            made("E_gene_expression_by_domain")

        a, b = str(args.domain_a), str(args.domain_b)
        status["DE_domains"] = [a, b]
        if a == b or a not in set(labels) or b not in set(labels):
            skipped("F_H_I_DE", "Choose two different saved domain IDs via --domain-a/--domain-b")
        else:
            focus = frame.loc[(labels == a) | (labels == b)]
            domain_map(focus, "Ours", args.out_dir / "F_selected_domain_locations",
                       args.dpi, args.point_size)
            made("F_selected_domain_locations")
            de = de_test(norm, genes, labels == a, labels == b)
            de.insert(1, "display_name", de.gene.map(lambda g: display.get(key(g), g)))
            de.to_csv(args.out_dir / "F_DE_domain_{}_vs_{}.csv".format(a, b), index=False)
            status["volcano_display"] = volcano(
                de, a, b, args.out_dir / "F_DE_volcano", args.dpi,
                args.volcano_y_max, args.volcano_label_genes, display)
            made("F_DE_volcano")
            domains = list(args.comparison_domains)
            if not set(domains).issubset(set(labels)):
                skipped("F_three_domain_comparison", "Comparison domains absent from saved predictions")
            else:
                triplet = comparison_triptych(norm, genes, frame, labels, domains,
                                             de, args.comparison_genes,
                                             status["volcano_display"]["display_y_max"],
                                             args.dpi, args.out_dir, display)
                status["comparison_triptych"] = triplet
                if "skipped" in triplet:
                    skipped("F_three_domain_comparison", triplet["skipped"])
                else:
                    for name in triplet["outputs"][:3]:
                        made(name)
            if deg_heatmap(norm, genes, labels, a, b, de, args.out_dir / "H_DE_heatmap",
                           args.dpi, args.seed, display):
                made("H_DE_heatmap")
            else:
                skipped("H_DE_heatmap", "No DEGs meet FDR < 0.05 and |log2FC| >= 2")
            if args.gmt:
                if not args.gmt.is_file():
                    raise FileNotFoundError(args.gmt)
                status["inputs"]["gmt"] = str(args.gmt.resolve())
                if go_enrichment(de, args.gmt, args.out_dir, args.dpi, a, b):
                    made("G_GO_enrichment")
                else:
                    skipped("G_GO_enrichment", "No enriched gene set passes BH FDR < 0.05")
            else:
                skipped("G_GO_enrichment", "Supply an appropriate GO .gmt file")
            selected_genes = []
            for target in args.violin_domains:
                if target not in set(labels):
                    continue
                ranked = de_test(norm, genes, labels == target, labels != target)
                ranked.insert(1, "display_name", ranked.gene.map(lambda g: display.get(key(g), g)))
                ranked.to_csv(args.out_dir / ("I_DE_domain_{}_vs_rest.csv".format(target)), index=False)
                selected_genes.extend(ranked[(ranked.log2FC > 0) & (ranked.fdr < .05)]
                                      .head(3).gene.tolist())
            selected_genes = list(dict.fromkeys(selected_genes))
            if selected_genes and violin(norm, genes, labels, selected_genes,
                                         args.out_dir / "I_top_DEGs_stacked_violin", args.dpi, display):
                made("I_top_DEGs_stacked_violin")
                status["I_genes"] = selected_genes
                status["I_display_names"] = [display.get(key(g), g) for g in selected_genes]
            else:
                skipped("I_top_DEGs_stacked_violin", "No significant domain-vs-rest positive DEGs")

        if args.histology:
            if not args.histology.is_file():
                raise FileNotFoundError(args.histology)
            image = plt.imread(args.histology)
            fig, ax = plt.subplots(figsize=(5, 4.4))
            ax.imshow(image)
            ax.axis("off")
            save(fig, args.out_dir / "J_original_HE_image", args.dpi)
            made("J_original_HE_image")
            status["inputs"]["histology"] = str(args.histology.resolve())
        else:
            skipped("J_original_HE_image", "Supply original H&E with --histology")
        skipped("K_telepathology", "Requires independent telepathology labels; saved model output is in B")
    finally:
        (args.out_dir / "analysis_status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
