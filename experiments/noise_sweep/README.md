# HA-site random-mutation noise sweep

Files: `noise_sweep.py` (the sweep), `run_noise_sweep.sbatch` (SLURM job),
`serine_proteases.fasta` (6 human trypsin-like serine proteases, from the paper repo's
`uniprot_human_full.fasta`), `active_sites.json` (UniProt catalytic His/Asp/Ser, 1-based,
from the repo's `prot_activeSite.pkl`).

## Run on Sherlock

```bash
mkdir -p /oak/stanford/groups/rbaltman/rnarain7/noise_sweep/logs
# copy the four files into that folder, then:
cd /oak/stanford/groups/rbaltman/rnarain7/noise_sweep
PY=/oak/stanford/groups/rbaltman/rnarain7/envs/ha_research/bin/python
$PY -c "import esm, pwlf, scipy, torch; print('deps ok')"   # needs fair-esm, pwlf, scipy
$PY noise_sweep.py --download-only    # only if ESM-2 650M weights aren't cached yet
sbatch run_noise_sweep.sbatch
```

Quick smoke test first (one protein, one level, ~2 min):
`$PY noise_sweep.py --out smoke --proteins P07477 --levels 10 --seeds 1`

## Design

- Noise levels 5, 10, 20, 30, 50, 70% (`--levels`), 5 seeds each (`--seeds`). Positions are chosen
  uniformly at random; each becomes a different random standard amino acid.
- 0% is also run with 5 pwlf seeds: pwlf's fit is stochastic, so those rows give the noise floor
  from the fitting alone.
- The HA-site method is copied unchanged from `ESM_ha_loc.py`, including the `floor(break*n) - 1` cutoff.

## Output: `results/noise_sweep.csv`, one row per variant

All residue numbers are 1-based.

| column | meaning |
|---|---|
| `layer`, `wt_layer`, `layer_same` | convergence layer of the variant / unmutated sequence |
| `n_ha`, `wt_n_ha` | number of HA sites |
| `jaccard`, `jaccard_chance` | HA-set overlap with unmutated, and expected overlap for a random set of the same size |
| `wt_ha_recall`, `wt_ha_recall_pm2` | fraction of unmutated HA sites still HA (exact, or within ±2 residues) |
| `wt_ha_mutated_frac` | fraction of unmutated HA residues that were themselves mutated |
| `*_at_wt_layer` | the same, reading HA sites at the unmutated convergence layer (separates layer switching from site loss) |
| `spearman_wt_layer(_unmutated)` | rank correlation of the attention vector with unmutated at that layer (all positions / unmutated positions only) |
| `ha_min_dist_to_active_median`, `ha_frac_within3_active`, `active_is_ha_frac` | sequence distance from HA sites to the catalytic triad |
| `ha_sites`, `mutated_positions`, … | the raw sets |

`results/heatmaps/*.npz` holds the 33×n attention matrix, per-layer θ, mutated positions and sequence for every variant.
