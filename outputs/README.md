# Outputs (generated locally — not committed)

Training writes artifacts here. Regenerate with:

```bash
./run_train.sh --experiment all --epochs 20
```

| Path | Description |
|------|-------------|
| `experiment_results.json` | Metrics for all runs |
| `*.pt` | Best-checkpoint weights per run |
| `figures/` | Qualitative panels from `python scripts/eval.py` |

These files are gitignored except this README.
