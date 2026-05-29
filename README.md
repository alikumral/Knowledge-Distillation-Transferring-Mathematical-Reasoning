# Knowledge Distillation of Mathematical Reasoning (Llama-3.1-8B → Llama-3.2-1B)

CS 455 term project — Category (D) Efficiency/Optimization.
**Ali Kumral (30586), Revna Demirkale (32056).**

We distill multi-step math reasoning from **Llama-3.1-8B-Instruct** (teacher) into
**Llama-3.2-1B-Instruct** (student) by fine-tuning the student with **QLoRA** on
**rejection-sampled teacher chain-of-thought (CoT)** traces over GSM8K, and we test
whether teacher reasoning beats the dataset's own gold solutions and whether gains
generalize to out-of-distribution MATH problems.

See [plan.md](plan.md) for the full design, hypotheses, timeline, and risks.

---

## Pipeline

| Stage | Module / script | Notebook | Output |
|---|---|---|---|
| 1. Teacher CoT generation | `generate.py` / `run_generate.py` | `01_teacher_generate.ipynb` | `traces/teacher_traces.jsonl` |
| 2. Rejection sampling + SFT build | `reject.py` / `run_reject.py` | `02_reject_and_build.ipynb` | `sft/{accepted,gold,teacher_1,teacher_3}.jsonl` |
| 3. QLoRA fine-tuning | `train.py` / `run_train.py` | `03_train_qlora.ipynb` | `adapters/{condition}/seed{n}/` |
| 4. Evaluation + analysis | `evaluate.py` / `run_eval.py` | `04_evaluate.ipynb` | `results/predictions/*.jsonl` |

**Comparison conditions:** (i) zero-shot 1B · (ii) 1B + gold CoT · (iii) 1B + 1 teacher CoT ·
(iv) 1B + 3 teacher CoT · (v) 8B teacher (upper bound).

---

## Setup (Colab Pro, GPU runtime)

1. **Request gated access** to both models on Hugging Face and accept the license:
   - `meta-llama/Llama-3.1-8B-Instruct`
   - `meta-llama/Llama-3.2-1B-Instruct`
2. Create an HF token (**write** scope) and add it as a Colab **secret** named `HF_TOKEN`.
3. Open **`notebooks/00_setup.ipynb`**, set `REPO_URL`, and run it. It installs the
   package, logs into HF, mounts Drive, and smoke-tests both models in 4-bit.

Artifacts are written to Google Drive via the `MATHDISTILL_HOME` environment variable
(default on Colab: `/content/drive/MyDrive/math-distillation`) so they survive
disconnects. Every stage is resumable.

### Local install (for development / running the tests; no GPU needed)

```bash
pip install -e ".[dev]"
pytest -q          # 57 tests for the pure-Python core (prompts, answers, data, metrics)
```

> The GPU stages (`generate`, `train`, `evaluate`) lazy-import torch/transformers,
> so the package imports and the test suite run fine on a CPU-only machine.

---

## Run order (end-to-end)

Run the notebooks `00 → 04` in order, or the CLI equivalents:

```bash
# Stage 1 — teacher generation (~6h on T4; resumable)
python scripts/run_generate.py --config configs/teacher_gen.yaml

# Stage 2 — rejection sampling + SFT datasets (CPU)
python scripts/run_reject.py

# Stage 3 — train one (condition, seed); repeat across conditions/seeds
python scripts/run_train.py --config configs/train_teacher_1.yaml --seed 0
python scripts/run_train.py --config configs/train_teacher_3.yaml --seed 0
python scripts/run_train.py --config configs/train_gold.yaml      --seed 0

# Stage 4 — evaluate each condition on GSM8K test + MATH-200
python scripts/run_eval.py --model-id meta-llama/Llama-3.2-1B-Instruct --name zeroshot
python scripts/run_eval.py --model-id meta-llama/Llama-3.2-1B-Instruct --adapter adapters/teacher_1/seed0 --name teacher_1_seed0
python scripts/run_eval.py --model-id meta-llama/Llama-3.1-8B-Instruct --name teacher
```

**Minimum Viable Result** (answers all hypotheses): conditions i, ii, iii, iv, v at
seed 0. Extra seeds and the LoRA-rank ablation are upgrades (plan.md §10).

---

## Configuration

All hyperparameters live in `configs/` (no magic numbers in code):

| File | Purpose |
|---|---|
| `data.yaml` | dataset ids; frozen MATH-200 OOD subset |
| `teacher_gen.yaml` | teacher model, 4-bit quant, sampling (k, temperature) |
| `train_teacher_1.yaml`, `train_teacher_3.yaml`, `train_gold.yaml` | LoRA + SFT per condition |
| `eval.yaml` | greedy decoding, latency/VRAM measurement |

---

## Data

- **GSM8K** (`openai/gsm8k`, `main`): train → teacher generation; test → final eval only.
- **MATH** (`qwedsacf/competition_math`): a frozen, level-stratified **MATH-200** OOD subset.

Both are public, license-clean, and PII-free. We release the cleaned teacher-CoT
traces as a derivative HF dataset (link added on completion).

---

## Reproducibility

- `seed_everything` covers Python/NumPy/torch/transformers; eval uses greedy decoding.
- Each training run writes `run_meta.json` (config + seed).
- **Pin versions** in `requirements.txt` after `00_setup` succeeds (TRL/transformers
  APIs drift — the version-sensitive SFT fields are flagged in `train.py`).

## Repository layout

```
src/mathdistill/   core library (prompts, answers, data, models, generate, reject, train, evaluate, metrics)
configs/           YAML configs per stage/condition
notebooks/         00_setup + 01–04 Colab drivers
scripts/           CLI equivalents of the notebooks
tests/             pure-Python unit + import tests
results/           committed final tables/figures/predictions
plan.md            full project plan
```

## AI-usage disclosure

Per course policy, our use of AI assistants in code and writing will be documented
in the final report.
