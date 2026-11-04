# HA-Site Random-Mutation Noise Sweep

Tests whether High-Attention (HA) sites remain stable when protein sequences are randomly mutated at varying levels.

## Dataset

6 human serine proteases (trypsin-like, PF00089):
- P07477 (Trypsin-1)
- P08246 (Neutrophil elastase)  
- P08311 (Cathepsin G)
- P00742, P07288, P17538 (other family members)

## Experimental Design

- **Noise levels**: 0%, 5%, 10%, 20%, 30%, 50%, 70% residues randomly swapped for different amino acids
- **Replicates**: 5 random seeds per level
- **0% baseline**: 5 replicates with stochastic PWLF fitting noise floor
- **Total runs**: 210 (6 proteins × 7 levels × 5 seeds)

Position selection: Uniformly at random; each mutated position becomes a different random standard amino acid.

## Key Findings: HA Sites Have a Stability Cliff

HA sites are robust to moderate sequence perturbation but collapse catastrophically at high divergence:

| Noise Level | Jaccard Similarity | HA Recall | Layer Stability | Interpretation |
|---|---|---|---|---|
| 0% | 1.000 | 1.000 | 100% | Perfect baseline |
| 5% | 0.983 ± 0.063 | 1.000 ± 0.000 | 100% | Essentially unchanged |
| 10% | 0.906 ± 0.212 | 0.950 ± 0.153 | 100% | Minor degradation |
| 20% | 0.894 ± 0.242 | 0.922 ± 0.204 | 100% | Gradual loss |
| 30% | 0.714 ± 0.369 | 0.761 ± 0.335 | 90% | **Stability threshold** |
| 50% | 0.153 ± 0.225 | 0.161 ± 0.234 | 43% | **Catastrophic collapse** |
| 70% | 0.039 ± 0.150 | 0.039 ± 0.150 | 7% | Complete failure |

### What the Metrics Mean

- **Jaccard similarity**: Fraction of overlap between WT and mutant HA sites (0=no overlap, 1=perfect match)
- **HA recall**: Fraction of wildtype HA sites recovered in the mutant (how many WT hotspots survive)
- **Layer stability**: Percentage of proteins where convergence layer stayed at layer 10

### Critical Insight

**Between 30% and 50% sequence divergence, HA sites transition from "mostly stable" to "nearly lost."**

- Up to 30%: HA sites are surprisingly robust; most remain identifiable
- At 50%: Median HA site overlap drops to 15% — HA sites are essentially destroyed
- At 70%: Only 4% overlap remains; the attention pattern has fundamentally shifted

## Biological Interpretation

Random mutations at ~30% divergence (roughly 6-9 amino acid changes per 20-residue window) still preserve the HA site signature that ESM-2 learns. But at 50%+ divergence, the "functional core" that the model identifies becomes unrecoverable.

This suggests HA sites encode **structural/functional constraints** — they're not just sequence artifacts.

## Implications for Biosecurity

Random mutation is a poor model for protein evolution. The next phase tests **ProteinMPNN-generated sequences**, which should better preserve fold structure. If HA sites survive realistic mutations better than random noise, that's evidence the model captures true functional determinants.

## Files

- `noise_sweep.py` - Main analysis script
- `run_noise_sweep.sbatch` - SLURM job submission script
- `results/noise_sweep.csv` - Results table (210 rows, one per variant)
- `results/heatmaps/*.npz` - Full attention matrices per variant (33 layers × protein length)
- `logs/` - SLURM output logs

## Output Columns (in results/noise_sweep.csv)

| Column | Meaning |
|--------|---------|
| `protein` | UniProt ID |
| `length` | Protein sequence length |
| `noise_pct` | Mutation level (%) |
| `seed` | Random seed for reproducibility |
| `n_mutated` | Number of residues actually mutated |
| `wt_layer` | Convergence layer for wildtype |
| `layer` | Convergence layer for mutant |
| `layer_same` | 1 if layer stayed at 10, 0 if shifted |
| `jaccard` | HA site overlap with wildtype |
| `wt_ha_recall` | Fraction of WT HA sites recovered |
| `wt_ha_mutated_frac` | Fraction of WT HA residues that were themselves mutated |
| `ha_min_dist_to_active_median` | Distance from HA to catalytic triad |
| `ha_sites`, `wt_ha_sites` | Positions of HA sites |

## Run on Sherlock

```bash
mkdir -p /oak/stanford/groups/rbaltman/rnarain7/noise_sweep/logs
cd /oak/stanford/groups/rbaltman/rnarain7/noise_sweep
PY=/oak/stanford/groups/rbaltman/rnarain7/envs/ha_research/bin/python

# Check dependencies
$PY -c "import esm, pwlf, scipy, torch; print('deps ok')"

# Quick smoke test (one protein, one level, ~2 min)
$PY noise_sweep.py --out smoke --proteins P07477 --levels 10 --seeds 1

# Full run
$PY noise_sweep.py --download-only    # only if ESM-2 weights aren't cached
sbatch run_noise_sweep.sbatch
```

## Design Notes

- The HA-site method is unchanged from `ESM_ha_loc.py`, including `floor(break*n) - 1` cutoff
- All residue numbers are 1-based
- `heatmaps/*.npz` contains: 33×n attention matrix, per-layer θ, mutated positions, and sequence for every variant
