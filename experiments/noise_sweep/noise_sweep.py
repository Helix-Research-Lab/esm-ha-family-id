"""Random-mutation noise sweep for ESM-2 high-attention (HA) sites.

For each protein: take the unmutated sequence, then substitute a random x% of
residues (each with a different random standard amino acid) over several seeds,
and measure how the convergence layer and HA sites respond.

HA-site method is copied from Helix-Research-Lab/ESM_highAttention
(ESM_matrices.py + ESM_ha_loc.py): mean-pool heads, drop BOS/EOS, column sum,
divide by max, round to 2 dp, pwlf 2-segment fit on the descending-sorted
vector against x = linspace(0, 1, n), convergence layer = theta closest to 90,
HA = sorted_desc[:floor(break * n) - 1].

Indexing: internally 0-based. Every residue number written to the CSV is
1-based (UniProt numbering), as are the active sites in active_sites.json.

Usage:
    python noise_sweep.py --download-only          # on a login node, caches weights
    python noise_sweep.py --out results            # the sweep itself
"""
import argparse
import csv
import json
import math
import os
import time
import zlib

import numpy as np
import pwlf
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
AA = "ACDEFGHIKLMNPQRSTVWY"
N_LAYERS = 33
FIELDS = [
    "protein", "length", "noise_pct", "seed", "n_mutated",
    "wt_layer", "layer", "layer_same",
    "wt_n_ha", "n_ha", "jaccard", "jaccard_chance", "wt_ha_recall", "wt_ha_recall_pm2",
    "wt_ha_mutated_frac",
    "n_ha_at_wt_layer", "jaccard_at_wt_layer", "wt_ha_recall_at_wt_layer",
    "spearman_wt_layer", "spearman_wt_layer_unmutated",
    "ha_min_dist_to_active_median", "ha_frac_within3_active", "active_is_ha_frac",
    "ha_sites", "wt_ha_sites", "ha_sites_at_wt_layer", "mutated_positions", "seconds",
]


def read_fasta(path):
    seqs, name = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split("|")[1] if "|" in line else line[1:].split()[0]
                seqs[name] = ""
            elif name:
                seqs[name] += line
    return seqs


def attention_heatmap(model, batch_converter, name, seq, device):
    """33 x n matrix of normalized column sums, as in ESM_ha_loc.make_normalized_heatmaps."""
    _, _, tokens = batch_converter([(name, seq)])
    with torch.no_grad():
        out = model(tokens.to(device), repr_layers=[], need_head_weights=True)
    attn = out["attentions"][0].float().mean(dim=1)  # (layers, T, T), heads mean-pooled
    attn = attn[:, 1:-1, 1:-1].cpu()
    heatmap = []
    for layer in range(N_LAYERS):
        col = torch.sum(attn[layer], dim=0)
        heatmap.append(np.round((col / max(col)).numpy(), 2))
    return np.array(heatmap)


def fit_layers(heatmap, pwlf_seed):
    """Per-layer HA sites and theta, as in ESM_ha_loc.get_LoC_HA. Returns (layer, ha_by_layer, thetas)."""
    n = heatmap.shape[1]
    x = np.linspace(0, 1, n)
    ha_by_layer, thetas = [], []
    for layer in range(N_LAYERS):
        vec = heatmap[layer, :]
        desc = np.argsort(vec)[::-1]
        try:
            fit = pwlf.PiecewiseLinFit(x, vec[desc], seed=pwlf_seed)
        except TypeError:  # older pwlf without a seed argument
            np.random.seed(pwlf_seed)
            fit = pwlf.PiecewiseLinFit(x, vec[desc])
        breaks = fit.fit(2)
        ha_by_layer.append(sorted(int(i) for i in desc[: math.floor(breaks[1] * n) - 1]))
        m1, m2 = float(fit.slopes[0]), float(fit.slopes[1])
        thetas.append(math.degrees(math.atan((m2 - m1) / (1 + m2 * m1))))
    layer = int(np.argmin([abs(t - 90) for t in thetas]))
    return layer, ha_by_layer, thetas


def mutate(seq, pct, rng):
    n = len(seq)
    k = int(round(n * pct / 100))
    pos = np.sort(rng.choice(n, size=k, replace=False)) if k else np.array([], dtype=int)
    s = list(seq)
    for p in pos:
        s[p] = rng.choice([a for a in AA if a != seq[p]])
    return "".join(s), [int(p) for p in pos]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")


def recall(ref, new, tol=0):
    if not ref:
        return float("nan")
    return float(np.mean([any(abs(r - q) <= tol for q in new) for r in ref]))


def chance_jaccard(na, nb, n):
    inter = na * nb / n
    union = na + nb - inter
    return inter / union if union else float("nan")


def ones(idx):
    return ";".join(str(i + 1) for i in idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=os.path.join(HERE, "serine_proteases.fasta"))
    ap.add_argument("--active-sites", default=os.path.join(HERE, "active_sites.json"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--levels", default="5,10,20,30,50,70", help="noise percentages; 0 is always run")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--proteins", default="", help="comma-separated subset of accessions")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if args.download_only:
        print("ESM-2 650M weights are cached at", torch.hub.get_dir())
        return
    model = model.eval().to(args.device)
    batch_converter = alphabet.get_batch_converter()

    seqs = read_fasta(args.fasta)
    if args.proteins:
        seqs = {p: seqs[p] for p in args.proteins.split(",")}
    active = {p: [a - 1 for a in v] for p, v in json.load(open(args.active_sites)).items()}
    levels = [float(x) for x in args.levels.split(",")]

    os.makedirs(os.path.join(args.out, "heatmaps"), exist_ok=True)
    csv_path = os.path.join(args.out, "noise_sweep.csv")
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            done = {(r["protein"], float(r["noise_pct"]), int(r["seed"])) for r in csv.DictReader(f)}
    new_file = not os.path.exists(csv_path)
    fout = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
    print(f"device={args.device} proteins={list(seqs)} levels={levels} seeds={args.seeds}", flush=True)

    for prot, wt_seq in seqs.items():
        n = len(wt_seq)
        act = active.get(prot, [])
        wt = None
        # 0% with seeds > 0 re-fits the unmutated sequence with a different pwlf seed:
        # the noise floor of the fitting step itself.
        runs = [(0.0, s) for s in range(args.seeds)] + [(lv, s) for lv in levels for s in range(args.seeds)]
        for pct, seed in runs:
            if wt is not None and (prot, pct, seed) in done:
                continue
            t0 = time.time()
            # zlib.crc32 is stable across runs, unlike hash()
            rng = np.random.default_rng(zlib.crc32(f"{prot}|{pct}|{seed}".encode()))
            seq, mut = mutate(wt_seq, pct, rng)
            hm = attention_heatmap(model, batch_converter, prot, seq, args.device)
            layer, ha_by_layer, thetas = fit_layers(hm, pwlf_seed=seed)
            ha = ha_by_layer[layer]
            np.savez_compressed(os.path.join(args.out, "heatmaps", f"{prot}_{pct:g}_{seed}.npz"),
                                heatmap=hm, thetas=np.array(thetas), mutated=np.array(mut), seq=seq)
            if wt is None:
                wt = dict(hm=hm, layer=layer, ha=ha)
                if (prot, pct, seed) in done:
                    continue
            mut_set = set(mut)
            ha_wl = ha_by_layer[wt["layer"]]
            keep = [i for i in range(n) if i not in mut_set]
            dists = [min(abs(h - a) for a in act) for h in ha] if act else []
            row = dict(
                protein=prot, length=n, noise_pct=pct, seed=seed, n_mutated=len(mut),
                wt_layer=wt["layer"], layer=layer, layer_same=int(layer == wt["layer"]),
                wt_n_ha=len(wt["ha"]), n_ha=len(ha),
                jaccard=jaccard(wt["ha"], ha), jaccard_chance=chance_jaccard(len(wt["ha"]), len(ha), n),
                wt_ha_recall=recall(wt["ha"], ha), wt_ha_recall_pm2=recall(wt["ha"], ha, tol=2),
                wt_ha_mutated_frac=np.mean([h in mut_set for h in wt["ha"]]) if wt["ha"] else float("nan"),
                n_ha_at_wt_layer=len(ha_wl), jaccard_at_wt_layer=jaccard(wt["ha"], ha_wl),
                wt_ha_recall_at_wt_layer=recall(wt["ha"], ha_wl),
                spearman_wt_layer=spearmanr(wt["hm"][wt["layer"]], hm[wt["layer"]])[0],
                spearman_wt_layer_unmutated=spearmanr(wt["hm"][wt["layer"], keep], hm[wt["layer"], keep])[0],
                ha_min_dist_to_active_median=float(np.median(dists)) if dists else float("nan"),
                ha_frac_within3_active=float(np.mean([d <= 3 for d in dists])) if dists else float("nan"),
                active_is_ha_frac=float(np.mean([a in set(ha) for a in act])) if act else float("nan"),
                ha_sites=ones(ha), wt_ha_sites=ones(wt["ha"]), ha_sites_at_wt_layer=ones(ha_wl),
                mutated_positions=ones(mut), seconds=round(time.time() - t0, 1),
            )
            writer.writerow(row)
            fout.flush()
            print(f"{prot} {pct:g}% seed{seed}: layer {layer} (wt {wt['layer']}), HA {ones(ha)} "
                  f"(wt {ones(wt['ha'])}), jaccard {row['jaccard']:.2f}, {row['seconds']}s", flush=True)
    fout.close()
    print("done:", csv_path)


if __name__ == "__main__":
    main()
