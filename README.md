# ESM-2 HA Family Identification

Testing whether High-Attention (HA) sites encode stable, fold-specific functional signatures that can identify proteins across sequence divergence.

## Project Structure

```
experiments/
├── noise_sweep/
│   ├── noise_sweep.py
│   ├── run_noise_sweep.sbatch
│   ├── results/
│   │   ├── noise_sweep.csv
│   │   └── heatmaps/ (attention matrices per variant)
│   ├── logs/
│   └── README.md
└── proteinmpnn_sweep/
    ├── results/
    └── README.md (placeholder - coming soon)

data/
├── serine_proteases.fasta
└── active_sites.json
```

## Experiments

### 1. Noise Sweep (Completed)

Random mutation test: Do HA sites survive random amino acid changes?

**Key finding**: HA sites stable up to ~30% sequence divergence, collapse at 50%+.

See `experiments/noise_sweep/README.md` for details.

### 2. ProteinMPNN Sweep (In Progress)

Realistic mutation test using ProteinMPNN-generated sequences instead of random noise.

## Next Steps

- Run ProteinMPNN on serine proteases at varying divergence levels
- Compare HA site robustness under realistic vs. random mutations
- Extend analysis to other protein families
