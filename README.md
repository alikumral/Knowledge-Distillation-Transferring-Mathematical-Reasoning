# Knowledge Distillation of Mathematical Reasoning (Llama-3.1-8B → Llama-3.2-1B)

CS 455 term project — Category (D) Efficiency / Optimization.
**Ali Kumral (30586), Revna Demirkale (32056).**

We distill multi-step math reasoning from **Llama-3.1-8B-Instruct** (teacher) into
**Llama-3.2-1B-Instruct** (student) by fine-tuning the student with **QLoRA** on
**rejection-sampled teacher chain-of-thought (CoT)** traces over GSM8K. We test four
hypotheses: whether distillation works (H1), whether teacher reasoning beats the
dataset's own gold solutions (H2), whether more teacher traces help (H3), and
whether gains transfer out-of-distribution to MATH (H4).

- **Full write-up:** [`report/report.tex`](report/report.tex) (LaTeX) / [`report/report_draft.md`](report/report_draft.md).
- **Design, hypotheses, timeline, risks:** [`plan.md`](plan.md).

---

## Results (headline)

GSM8K test (1,319 problems) and MATH-167 OOD; accuracy in %. Trained conditions show
mean ± std over 3 seeds; baselines are deterministic (greedy).

| Condition | GSM8K | MATH-167 |
|---|---|---|
| (i) Zero-shot 1B | 38.06 | 33.53 |
| (ii) Gold CoT | 33.74 | 20.96 |
| **(iii) Teacher-1** | **46.93 ± 0.11** | 29.14 ± 1.98 |
| (iv) Teacher-3 | 44.33 ± 0.77 | 26.15 ± 0.56 |
| (v) 8B teacher | 82.87 | 53.89 |

- **H1 (distillation works): ✓** +8.9 pp over zero-shot, McNemar *p* < 10⁻⁹.
- **H2 (teacher ≫ gold): ✓** +13 pp, *p* < 10⁻¹⁷ — teacher reasoning beats gold labels.
- **H3 (more traces help): ✗ reversed** — Teacher-1 beats Teacher-3 (≈6σ across seeds).
- **H4 (OOD transfer): ✗** — no improvement on MATH.
- **Efficiency:** the 1B student uses **5.2× less peak VRAM** than the teacher
  (1.13 vs 5.87 GB); latency advantage is eroded by verbosity inheritance.

---

## Quick start (Colab — recommended)

The entire project runs from the single notebook **[`project.ipynb`](project.ipynb)**.

1. **Request gated access** on Hugging Face and accept the license for both:
   - `meta-llama/Llama-3.1-8B-Instruct`
   - `meta-llama/Llama-3.2-1B-Instruct`
2. Create an HF token (**write** scope) and add it as a Colab **secret** named `HF_TOKEN`.
3. Open `project.ipynb` in Colab (GPU runtime; we used an **A100**, but a T4 works
   more slowly). Run the **MUST-RUN-FIRST** cell — it clones the repo, installs deps,
   mounts Drive, logs into HF, and prints a setup summary. Re-run it after any
   restart (it also `git pull`s and clears stale module caches).
4. Run the stage cells in order (1 → 4). Each stage writes artifacts to Google Drive
   under `MATHDISTILL_HOME`, so a disconnect never loses completed work.

> All heavy artifacts (traces, adapters, predictions) live on Google Drive via the
> `MATHDISTILL_HOME` environment variable (set in the MUST-RUN-FIRST cell), **not**
> in the repo. Every stage is resumable.

---

## Pipeline

| Stage | Library module | What it does | Output (under `MATHDISTILL_HOME`) |
|---|---|---|---|
| 1. Teacher generation | `generate.py` | Llama-3.1-8B (4-bit) samples CoT over GSM8K train, **3 passes** (1 trace/problem each), resumable | `traces/teacher_traces.jsonl` |
| 2. Rejection sampling | `reject.py` | keep traces matching gold; build SFT datasets | `sft/{accepted,gold,teacher_1,teacher_3}.jsonl` |
| 3. QLoRA fine-tuning | `train.py` | train the 1B student per condition/seed | `adapters/{condition}/seed{n}/` |
| 4. Evaluation + analysis | `evaluate.py`, `metrics.py` | GSM8K + MATH accuracy, CIs, McNemar, latency/VRAM, difficulty, verbosity | `results/predictions/*.jsonl` |

**Conditions:** (i) zero-shot 1B · (ii) gold CoT · (iii) 1 teacher CoT · (iv) up to 3
teacher CoT · (v) 8B teacher. Trained conditions use 3 seeds.

The `notebooks/00_setup.ipynb` … `04_evaluate.ipynb` files are an optional modular
alternative to `project.ipynb`; `scripts/run_*.py` are CLI equivalents (useful on
Kaggle or locally).

---

## CLI equivalents (optional)

```bash
# Stage 1 — teacher generation. Run once per pass: set `pass_id: 0/1/2` in the config.
python scripts/run_generate.py --config configs/teacher_gen.yaml

# Stage 2 — rejection sampling + build the 3 SFT datasets (CPU only)
python scripts/run_reject.py

# Stage 3 — train one (condition, seed); repeat for seeds 0,1,2
python scripts/run_train.py --config configs/train_teacher_1.yaml --seed 0
python scripts/run_train.py --config configs/train_teacher_3.yaml --seed 0
python scripts/run_train.py --config configs/train_gold.yaml      --seed 0

# Stage 4 — evaluate each condition (GSM8K test + MATH-167 + latency/VRAM)
python scripts/run_eval.py --model-id meta-llama/Llama-3.2-1B-Instruct --name zeroshot
python scripts/run_eval.py --model-id meta-llama/Llama-3.2-1B-Instruct --adapter adapters/gold/seed0      --name gold_seed0
python scripts/run_eval.py --model-id meta-llama/Llama-3.2-1B-Instruct --adapter adapters/teacher_1/seed0 --name teacher_1_seed0
python scripts/run_eval.py --model-id meta-llama/Llama-3.2-1B-Instruct --adapter adapters/teacher_3/seed0 --name teacher_3_seed0
python scripts/run_eval.py --model-id meta-llama/Llama-3.1-8B-Instruct --name teacher
```

---

## Local install (development / tests — no GPU needed)

```bash
pip install -e ".[dev]"
pytest -q     # pure-Python core: prompts, answers, data, metrics, imports
```

The GPU stages (`generate`, `train`, `evaluate`) lazy-import torch/transformers, so
the package imports and the test suite run on a CPU-only machine.

---

## Configuration

All hyperparameters live in `configs/` (no magic numbers in code):

| File | Purpose |
|---|---|
| `data.yaml` | dataset ids; frozen MATH OOD subset (167 problems, seed-stratified) |
| `teacher_gen.yaml` | teacher model, 4-bit quant, sampling (temperature/top-p), `pass_id`, batch size |
| `train_teacher_1.yaml`, `train_teacher_3.yaml`, `train_gold.yaml` | LoRA + SFT per condition (3 seeds each) |
| `eval.yaml` | full eval: greedy decoding + latency/VRAM |
| `eval_seeds.yaml` | faster eval for extra seeds (GSM8K + MATH, latency off) |

---

## Data

- **GSM8K** (`openai/gsm8k`, `main`): train → teacher generation; test (1,319) → final eval only.
- **MATH** (`qwedsacf/competition_math`): a frozen, level-stratified **167-problem** OOD subset.

Both are public, license-clean, and PII-free.

---

## Reproducibility & known deviations

- `seed_everything` covers Python/NumPy/torch/transformers; evaluation uses greedy
  decoding (deterministic). Each training run writes `run_meta.json` (config + seed +
  TRL version).
- **Loss masking:** the installed TRL version removed the completion-only collator we
  targeted, so all conditions were trained with **full-sequence loss**. This is
  applied identically to every condition (fair comparison); see the report's
  Discussion/Limitations for the analysis of this confound.
- **Hardware:** generation/training/eval were run on a Colab A100; reported latency
  is A100-specific (the cross-model *ratio* is the meaningful quantity).
- **Pin versions:** for an exact rerun, freeze `requirements.txt` to the resolved
  versions from your working Colab session (TRL/transformers APIs drift).

---

## Repository layout

```
project.ipynb       single end-to-end Colab notebook (primary entry point)
src/mathdistill/    core library (prompts, answers, data, models, generate, reject, train, evaluate, metrics)
configs/            YAML configs per stage/condition
notebooks/          00_setup + 01–04 modular Colab drivers (alternative to project.ipynb)
scripts/            CLI equivalents of the stages
tests/              pure-Python unit + import tests
report/             report.tex (LaTeX) + report_draft.md
results/            figures/tables/predictions (artifacts live on Drive; figures inserted into the report)
plan.md             full project plan
```

---

## Released artifacts

- Trained LoRA adapter (best condition, Teacher-1): https://huggingface.co/alikumral/llama-3.2-1b-gsm8k-cot-distilled
- Cleaned teacher-CoT trace dataset: https://huggingface.co/datasets/alikumral/gsm8k-teacher-cot-traces

## AI-usage disclosure

Per course policy, our use of an AI assistant (for code structure, the `mathdistill`
modules, config/parameter choices, debugging, analysis tooling, and report drafting)
is documented in detail in **Appendix C of the report**. All experiments, decisions,
and reported numbers are our own.
