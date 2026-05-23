# ContamCal: Contamination Dose-Response Profiling via IRT

Code for the paper: *"Not All Models Are Inflated: Profiling Contamination Dose-Response Across Model Families"*

## Overview

ContamCal is a diagnostic framework that combines multi-signal contamination exposure mapping with contamination-augmented Item Response Theory (IRT) to profile how benchmark contamination affects LLM evaluation scores.

## Project Structure

- `contamination_injection/` — SFT-based contamination injection pipeline (LoRA fine-tuning at varying dosage levels)
- `detection_signals/` — Multi-signal contamination detection (embedding similarity, Min-K%++, self-critique)
- `irt_calibration/` — Contamination-augmented 2PL IRT model (Stan/MCMC) with baseline methods
- `analysis/` — Statistical analysis scripts (DiD, bootstrap, cross-seed stability)
- `simulation/` — IRT parameter recovery simulation
- `figures/` — Paper figure generation scripts
- `scripts/` — Evaluation shell scripts

## Setup

```bash
pip install -r requirements.txt
# Install CmdStan for IRT fitting
install_cmdstan
```

## Key Components

### 1. Contamination Injection (Stage 0)
```bash
python contamination_injection/prepare_sft_data.py --benchmark mmlu --dosage 0.25 --seed 42
python contamination_injection/train_sft.py --model Qwen/Qwen2.5-7B --data_path sft_data_d025_s42.jsonl
python contamination_injection/evaluate.py --model_path checkpoints/qwen7b_d025_s42 --benchmark mmlu
```

### 2. Detection Signal Computation (Stage 1)
```bash
python detection_signals/compute_signals.py --model_path checkpoints/qwen7b_d025_s42 --benchmark mmlu
python detection_signals/aggregate_rrf.py --signal_dir signals/ --output rrf_scores.csv
```

### 3. IRT Calibration (Stage 2)
```bash
python irt_calibration/fit_irt_2pl_gamma.py --data irt_input.csv --exposure rrf_scores.csv
```

### 4. Figure Generation
```bash
python figures/gen_fig_dose_response.py
python figures/gen_fig_domain_gamma.py
python figures/gen_fig_cross_seed_ci.py
```

## Citation

```bibtex
@inproceedings{anonymous2026contamcal,
  title={Not All Models Are Inflated: Profiling Contamination Dose-Response Across Model Families},
  author={Anonymous},
  booktitle={Proceedings of EMNLP 2026},
  year={2026}
}
```

## License

This code is released for research purposes under the MIT License.
