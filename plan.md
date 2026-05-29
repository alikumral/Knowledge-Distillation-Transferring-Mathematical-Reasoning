# Project Plan — Knowledge Distillation of Mathematical Reasoning

**Transferring Mathematical Reasoning from Llama-3.1-8B → Llama-3.2-1B via CoT + QLoRA**

| | |
|---|---|
| **Course / Track** | CS 455 — Category (D) Efficiency / Optimization |
| **Team** | Ali Kumral (30586), Revna Demirkale (32056) |
| **Compute** | Google Colab Pro (primary), Colab Free (fallback) |
| **Final deadline** | **Sunday, June 7, 2026, 23:59** (report + code zip + demo) |
| **Plan written** | 2026-05-28 |

> ⚠️ **Timeline note.** The proposal assumed a May 10 start (5 weeks). Today is **May 28** → we have **~10 days**. This plan is reorganized into a compressed, **priority-ordered** schedule. The full 5-condition × 3-seed design is preserved as the *target*, but we define a **Minimum Viable Result (MVR)** that already satisfies the proposal's core claims, then layer on extra seeds/ablations only if time allows. See [§10 Timeline](#10-compressed-timeline-may-28--june-7).

---

## 0. Proposal Status & Feedback Incorporated

**Status: ACCEPTED_WITH_MINOR_SUGGESTIONS** (instructor review). No redesign required. The three suggestions are folded into this plan as follows:

| Suggestion | Where addressed |
|---|---|
| 1. More comprehensive metrics — **qualitative** analysis of reasoning traces + latency/accuracy **trade-offs** | [§6 Stage 4](#stage-4--evaluation-srcevaluatepy-srcmetricspy) (qualitative trace study), [§7](#7-evaluation-deliverables-tablesfigures-for-the-report) (T3 trade-off, F4 qualitative comparison) |
| 2. Clearer strategy for **Colab** technical issues | [§9.1 Colab operational strategy](#91-colab-operational-strategy) (+ Kaggle as 2nd free GPU) |
| 3. **Flexible** timeline for fine-tuning/eval surprises | [§10](#10-compressed-timeline-may-28--june-7) MVR structure + explicit **de-scoping cut-lines** |

> Note for the report: the final write-up should briefly state that proposal feedback was received and how it was incorporated (good practice + shows responsiveness).

---

## 1. Goal & Hypotheses

**Goal.** Show that a 1B student fine-tuned on *teacher-generated, rejection-sampled* chain-of-thought (CoT) traces meaningfully closes the GSM8K accuracy gap to its 8B teacher, and that teacher traces beat training on the dataset's own gold CoTs.

**Hypotheses (what every experiment must answer):**

- **H1 — Distillation works.** Distilled 1B ≫ zero-shot 1B on GSM8K test (statistically significant).
- **H2 — Teacher signal > gold labels.** 1B trained on teacher CoTs > 1B trained on GSM8K gold CoTs (the *key* claim — isolates the value of teacher reasoning beyond labeled data).
- **H3 — More samples help.** 3 teacher CoTs/problem ≥ 1 teacher CoT/problem.
- **H4 — Real generalization, not contamination.** Gains transfer to out-of-distribution MATH; if MATH gains track GSM8K gains, improvement is genuine.

**Non-goals (explicitly out of scope):** logit/soft-label (Hinton) distillation, RLHF/DPO, training the teacher, multi-GPU training, models > 8B.

---

## 2. Experimental Design

Five conditions. Student-training conditions run across **3 seeds** (default `[0, 1, 2]`); non-trained conditions (i, v) are evaluated once with **greedy decoding** for determinism.

| # | Condition | Trained? | Training data | Role |
|---|-----------|----------|---------------|------|
| i | Zero-shot 1B | ✗ | — | Lower bound |
| ii | 1B + gold CoT | ✓ (×3 seeds) | GSM8K gold solutions | "No-teacher" control → **tests H2** |
| iii | 1B + 1 teacher CoT | ✓ (×3 seeds) | 1 accepted teacher trace/problem | **tests H1** |
| iv | 1B + 3 teacher CoT | ✓ (×3 seeds) | up to 3 accepted teacher traces/problem | **tests H3** |
| v | 8B teacher | ✗ | — | Upper bound |

**Decoding policy (must be identical across all conditions for fair comparison):**
- **Evaluation:** greedy (`temperature=0`, `do_sample=False`), `max_new_tokens=512`. Deterministic → seeds only vary *training*, not eval.
- **Teacher CoT generation:** sampling (`temperature≈0.7`, `top_p≈0.9`) to get *diverse* traces; k samples/problem.

---

## 3. Pipeline Architecture (4 stages)

```
                 GSM8K train (7,473)              GSM8K test (1,319)  +  MATH-200 (OOD)
                       │                                      │
        ┌──────────────▼──────────────┐                       │
   (1)  │  Teacher CoT Generation      │  Llama-3.1-8B, 4-bit  │
        │  k samples/problem, sampling │  → traces.jsonl       │
        └──────────────┬──────────────┘                       │
                       │                                       │
        ┌──────────────▼──────────────┐                       │
   (2)  │  Rejection Sampling          │  parse final answer   │
        │  keep traces == gold answer  │  → accepted.jsonl     │
        └──────────────┬──────────────┘                       │
                       │  build SFT sets (1/problem, 3/problem, gold)
        ┌──────────────▼──────────────┐                       │
   (3)  │  QLoRA Fine-tuning           │  Llama-3.2-1B, 4-bit  │
        │  TRL SFT, completion-only    │  → adapters/          │
        └──────────────┬──────────────┘                       │
                       │                                       │
        ┌──────────────▼───────────────────────────────────────▼──────┐
   (4)  │  Evaluation: exact-match acc (GSM8K + MATH), latency, VRAM,  │
        │  gen-length, per-difficulty error analysis, CIs             │
        └─────────────────────────────────────────────────────────────┘
```

**Design principle: logic lives in `src/`, notebooks are thin drivers.** Each stage reads/writes versioned JSONL artifacts on Google Drive, so a Colab disconnect never loses more than the current shard. Prompts and answer-extraction live in **one** module shared by generation, filtering, and eval — they must never diverge.

---

## 4. Repository Structure

```
math-distillation/
├── README.md                  # setup + end-to-end run instructions (graded deliverable)
├── plan.md                    # this file
├── pyproject.toml             # installable package: `pip install -e .`
├── requirements.txt           # pinned versions (see §5)
├── .gitignore                 # data/, outputs/, *.ckpt, .env, drive mounts
├── LICENSE
│
├── configs/                   # all hyperparameters externalized — no magic numbers in code
│   ├── data.yaml              # dataset ids, splits, MATH-200 sampling seed
│   ├── teacher_gen.yaml       # model id, k, temperature, max_tokens, batch size
│   ├── train_gold.yaml        # condition (ii)
│   ├── train_teacher_1.yaml   # condition (iii)
│   ├── train_teacher_3.yaml   # condition (iv)
│   └── eval.yaml              # decoding params, batch size, metric settings
│
├── src/mathdistill/
│   ├── __init__.py
│   ├── prompts.py             # SINGLE source of truth: CoT prompt + chat templates
│   ├── answers.py             # extract & normalize final answer (GSM8K #### + MATH \boxed{}); equality check
│   ├── data.py                # load GSM8K/MATH, build MATH-200, build SFT datasets per condition
│   ├── models.py              # 4-bit BnB load, LoRA config, tokenizer/chat-template setup
│   ├── generate.py            # stage 1: batched teacher sampling, resumable shards
│   ├── reject.py              # stage 2: rejection sampling + dataset stats
│   ├── train.py               # stage 3: QLoRA via TRL SFTTrainer, completion-only loss
│   ├── evaluate.py            # stage 4: run model on a split → predictions.jsonl
│   ├── metrics.py             # accuracy, bootstrap CIs, latency/VRAM, gen-length, difficulty buckets
│   └── utils.py               # seed_everything, JSONL io, Drive paths, logging, timers
│
├── notebooks/                 # Colab entry points (clone repo → import src → call stage)
│   ├── 00_setup.ipynb         # HF login, install, Drive mount, smoke test both models
│   ├── 01_teacher_generate.ipynb
│   ├── 02_reject_and_build.ipynb
│   ├── 03_train_qlora.ipynb   # parametrized by config path + seed
│   └── 04_evaluate.ipynb
│
├── scripts/                   # CLI equivalents of notebooks (for reproducibility / CI)
│   ├── run_generate.py
│   ├── run_reject.py
│   ├── run_train.py
│   └── run_eval.py
│
├── tests/                     # fast unit tests — answer extraction is the #1 source of silent bugs
│   ├── test_answers.py        # gold/pred parsing on tricky cases ($, commas, fractions, \boxed)
│   └── test_data.py           # SFT formatting, no train/test leakage
│
└── results/                   # COMMITTED: final tables, figures, raw predictions for report
    ├── tables/
    ├── figures/
    └── predictions/
```

**Artifacts that are NOT committed** (live on Google Drive, gitignored):
```
Drive/math-distillation/
├── data/            # downloaded datasets cache
├── traces/          # raw teacher generations (sharded JSONL)
├── sft/             # built SFT datasets per condition
├── adapters/        # LoRA checkpoints: {condition}/seed{n}/
└── logs/            # training logs, run metadata
```

---

## 5. Environment & Dependencies

**Runtime:** Colab Pro, GPU = **T4 (16 GB)** baseline; **L4 (24 GB)** if available (needed for the 3B fallback in Risk 2).

**Gated-model access — DO THIS FIRST (can take hours/days to approve):**
1. Request access to `meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.2-1B-Instruct` on Hugging Face.
2. Create an HF access token (write scope — we push adapters + dataset).
3. Store as a Colab secret `HF_TOKEN`; never hardcode. `huggingface_hub.login()` reads it.

**Core packages** (pin in `requirements.txt`; pin exact versions during `00_setup` once confirmed working, because TRL/transformers APIs drift):
```
torch                # use Colab's preinstalled CUDA build
transformers
datasets
accelerate
peft
trl                  # SFTTrainer + completion-only collator
bitsandbytes         # 4-bit NF4 quantization
sentencepiece
huggingface_hub
pyyaml
numpy, pandas, matplotlib
tqdm
```
> **Pin once it works.** Record the exact resolved versions in `requirements.txt` and the README so the grader's fresh Colab reproduces our run. TRL's completion-only API (`DataCollatorForCompletionOnlyLM` vs `assistant_only_loss`) differs by version — lock it.

**Optional speedup for stage 1:** `vllm` can dramatically accelerate teacher generation, *but* bitsandbytes-4-bit support in vLLM is fragile on T4. **Default = transformers batched generation** (reliable, matches the proposal's 6h budget). Treat vLLM as an optional optimization, not a dependency.

---

## 6. Stage-by-Stage Implementation Notes

### Stage 0 — Data (`src/data.py`, `configs/data.yaml`)
- GSM8K: `load_dataset("openai/gsm8k", "main")` → `train` (7,473), `test` (1,319). Gold answer = text after `####`.
- MATH OOD: `qwedsacf/competition_math`. Build **MATH-200**: stratified sample by `level` (1–5) and/or `type`, fixed `sampling_seed` in config so the OOD set is frozen and reproducible. Gold answer inside `\boxed{...}`.
- **Leakage guard:** GSM8K test is touched *only* at final eval. Add a `test_data.py` assertion that no test question appears in any SFT set.

### Stage 1 — Teacher generation (`src/generate.py`)
- Model: Llama-3.1-8B-Instruct, 4-bit NF4 + double quant, `bnb_4bit_compute_dtype=bfloat16`.
- For each train problem, draw **k samples** (start `k=4`; gives headroom to keep up to 3 correct for condition iv). ~7,473 × 4 ≈ 30k generations.
- Use the chat template + CoT prompt from `prompts.py`. Batch generation for throughput.
- **Resumability is mandatory** (Colab disconnects): write `traces/shard_{i}.jsonl` incrementally; on restart, skip problems already present. Each record: `{problem_id, sample_idx, prompt, completion, raw_text}`.

### Stage 2 — Rejection sampling (`src/reject.py`)
- Parse final answer from each completion via `answers.extract_pred`; compare to gold via `answers.is_correct` (numeric normalization: strip `$`, `,`, `%`, whitespace; tolerant int/float compare).
- Keep correct traces. Build three SFT datasets:
  - **gold** (ii): GSM8K's own CoT solution, 1/problem.
  - **teacher-1** (iii): 1 accepted trace/problem (pick first correct, or shortest — decide and document).
  - **teacher-3** (iv): up to 3 accepted traces/problem.
- **Report dataset stats** (goes in the paper): acceptance rate overall and by difficulty bucket, #problems with ≥1 correct trace, trace length distribution. *If teacher acceptance is low → trigger Risk 1 plan B.*

### Stage 3 — QLoRA fine-tuning (`src/train.py`)
- Base 1B in 4-bit NF4; LoRA adapters in bf16.
- LoRA: `r=16, alpha=32, dropout=0.05`, target = all attention + MLP proj (`q,k,v,o,gate,up,down`). (Rank 8 vs 16 is the planned ablation.)
- TRL `SFTTrainer` with **completion-only loss masking** (loss on the CoT+answer, not the prompt). Use the chat template; mask the user turn.
- Suggested start: `lr=2e-4`, cosine schedule, `epochs=2–3`, `per_device_batch=4–8` + grad accumulation, `max_seq_len=1024`, bf16, gradient checkpointing. **Finalize on the pilot run.**
- Save adapter to `adapters/{condition}/seed{n}/`. Log to `logs/`. One config + one seed = one run; the notebook loops seeds.
- **Pilot first:** train on ~200 examples, 50 steps, confirm loss decreases and a sample generation is sane before committing the full ~2h run.

### Stage 4 — Evaluation (`src/evaluate.py`, `src/metrics.py`)
- Load base 1B + adapter (or merge); greedy decode the eval split → `results/predictions/{condition}_seed{n}_{split}.jsonl`.
- **Primary metric:** GSM8K test exact-match accuracy.
- **Secondary:** MATH-200 accuracy; **latency** (median over N runs, batch size 1, T4); **peak VRAM** (`torch.cuda.max_memory_allocated`); **generation-length** distribution (verbosity inheritance check).
- **Stats:** per-condition accuracy = mean ± std over 3 seeds; **bootstrap 95% CIs** over the 1,319 test items; report H1/H2/H3 as CI comparisons (non-overlap or paired bootstrap), not just point estimates.
- **Error analysis (quantitative):** bucket problems by gold-CoT length (easy/medium/hard tertiles); accuracy per bucket per condition → table + figure. Sample & categorize failure modes (arithmetic slip, wrong setup, truncation, no final answer).
- **Qualitative trace study (per feedback #1):** beyond final-answer correctness, inspect the *reasoning quality* of the traces. Pick a fixed set of ~20–30 problems and lay teacher vs each student vs gold CoT **side by side**. Assess: (a) does the student reason like the teacher or just memorize answers? (b) coherence/faithfulness of intermediate steps; (c) cases where the student reaches the right answer via *wrong* reasoning (lucky) vs right reasoning; (d) where the student is *more concise yet still correct* (a positive trade-off). Tag a handful of illustrative examples for the report. This explains **where the student excels vs falls short**, not just how often.
- **Accuracy↔efficiency trade-off (per feedback #1):** explicitly relate accuracy to cost — accuracy vs latency and vs peak VRAM, 1B (each condition) vs 8B teacher. Frame as a Pareto view: "the distilled 1B recovers X% of the teacher's accuracy at ~1/8 the serving cost." This is the project's core deployment argument and must be stated as a trade-off, not two separate numbers.

---

## 7. Evaluation Deliverables (tables/figures for the report)

- **T1** Main results: accuracy ± CI for all 5 conditions, GSM8K + MATH-200.
- **T2** Efficiency: latency (s/problem) and peak VRAM, 1B vs 8B.
- **T3** Accuracy↔efficiency trade-off: accuracy vs cost (latency, VRAM) per condition — the deployment-value table.
- **F1** Accuracy by difficulty bucket (grouped bars).
- **F2** Generation-length distributions (teacher vs students vs gold).
- **F3** Sample-count effect: zero-shot → gold → teacher-1 → teacher-3 → teacher.
- **F4** Qualitative side-by-side: teacher vs student CoT on matched problems (2–3 illustrative cases incl. one student win, one loss).
- **Error taxonomy** table with example failures.
- **Qualitative findings** paragraph: where the student reasons well vs where it breaks, with examples.

These directly map to report sections: Method, Results, Error Analysis (quantitative + qualitative), OOD, Efficiency & Trade-offs, Limitations.

---

## 8. Reproducibility Checklist (graded — README must run end-to-end)

- [ ] `seed_everything(seed)` covers `random`, `numpy`, `torch`, CUDA, and `transformers.set_seed`.
- [ ] Every run records its config + git commit + resolved package versions to `logs/`.
- [ ] All hyperparameters in `configs/*.yaml`, zero magic numbers in code.
- [ ] Datasets pinned by HF revision where possible; MATH-200 frozen by `sampling_seed`.
- [ ] README: gated-model access steps, `HF_TOKEN` secret, exact run order, expected runtimes.
- [ ] Release: cleaned teacher-CoT dataset (HF) + best adapter (HF), linked in README.

---

## 9. Compute Budget (T4, parallelize across days/accounts)

| Task | Estimate |
|---|---|
| Teacher generation (~30k gens, 4-bit 8B) | ~6 h |
| Student fine-tune (~2 h × runs) | MVR: 3 runs ≈ 6 h · Full: up to 9 runs ≈ 18 h |
| Evaluation (GSM8K test + MATH-200, per condition) | ~1 h each |
| Pilot, retries, ablations buffer | ~10 h |

Colab Pro disconnects → **checkpoint everything to Drive**, design every stage to resume.

### 9.1 Colab operational strategy (per feedback #2)

Colab's limitations (session timeouts, GPU-quota exhaustion, idle disconnects, occasional library breakage) are the most likely *operational* failure mode. Concrete mitigations:

- **Everything resumes.** Stages 1–3 write incrementally to Google Drive; a killed session resumes from the last shard/checkpoint, never from scratch. Test the resume path *once* on purpose early on.
- **Persist immediately, not at the end.** Adapters → Drive every N steps (`save_steps`) and pushed to HF Hub at the end of each run; trace shards flushed continuously. Nothing important ever lives only in the VM's local disk.
- **Pin the environment.** Record exact resolved versions in `requirements.txt` after `00_setup` works, so a future session (or the grader's) doesn't get a broken transformers/TRL upgrade. A dependency surprise mid-deadline is the silent killer.
- **Watch the clock.** Keep runs ≤ ~2 h each (one fine-tune) so a disconnect costs at most one run. Note remaining GPU units; don't start a long job on a near-empty quota.
- **Spread the load.** Two teammates' accounts + Colab Free fallback multiply available GPU hours. Coordinate so we don't both burn quota on the same run.
- **Kaggle as a second free GPU (per announcement §7).** Kaggle Notebooks give **30 GPU-h/week** (T4×2 / P100) entirely separate from Colab's quota. Keep code as a pip-installable repo + CLI scripts (`scripts/run_*.py`) so the *same* code runs on Kaggle unchanged — our insurance if Colab quota runs dry mid-week.
- **Reproducible disconnect log.** Log each run's start/resume/finish to `logs/` so we can report total wall-clock honestly in the paper.

---

## 10. Compressed Timeline (May 28 → June 7)

> **MVR (Minimum Viable Result)** = conditions **i, iii, iv, v** at **1 seed** + condition **ii** at 1 seed. This alone answers H1, H2, H3, H4 and is a complete, honest CS455 result. Extra seeds + rank ablation are upgrades.

| Day | Date | Tasks | Exit criteria |
|---|---|---|---|
| 1 | Wed 5/28 | **Request Llama gated access NOW.** Repo skeleton, `prompts.py`, `answers.py` + `tests/`. `00_setup` smoke test. | Both models load in 4-bit on T4; answer-extraction tests pass. |
| 2 | Thu 5/29 | Zero-shot baselines (cond i, v) on a dev sample; lock prompts. Start teacher generation (k=4). | Baselines plausible; generation running + resumable. |
| 3 | Fri 5/30 | Finish teacher generation; rejection sampling + dataset stats; build all 3 SFT sets. | accepted.jsonl + 3 SFT datasets; acceptance rate known (Risk 1 check). |
| 4 | Sat 5/31 | Training pilot → finalize hyperparams. Train **iii, iv (seed 0)**. | 2 adapters trained; pilot validated. |
| 5 | Sun 6/1 | Train **ii (seed 0)**. Full GSM8K-test eval of i, ii, iii, iv, v. | **MVR complete** — H1/H2/H3 answerable. |
| 6 | Mon 6/2 | MATH-200 OOD eval (H4); latency + VRAM benchmark; gen-length. | All metrics for MVR collected. |
| 7 | Tue 6/3 | Per-difficulty error analysis; generate T1–T2, F1–F3. Start report (Method/Results). | Results section drafted with real numbers. |
| 8 | Wed 6/4 | **Upgrades if on track:** seeds 1–2 for ii/iii/iv; rank 8 vs 16 ablation. Else: deepen error analysis. | CIs over seeds, or richer analysis. |
| 9 | Thu 6/5 | Finish report (Related Work, Limitations incl. contamination, Future Work, AI-usage disclosure). | Full draft done. |
| 10 | Fri 6/6 | README end-to-end test on fresh Colab; push dataset + adapter to HF; figures polish; proofread. | Repo reproduces; deliverables linked. |
| — | Sat 6/7 | **Buffer + submit early** (PDF report + code zip). Do NOT wait for 23:59. | Submitted by afternoon. |

**Flex & de-scoping cut-lines (per feedback #3).** The schedule is deliberately elastic — fine-tuning and eval are exactly where surprises happen, so we cut from the *bottom up*, never from the MVR:

- **Protected (never cut):** conditions i, ii, iii, iv, v at **1 seed** + GSM8K-test eval + MATH-200 (Days 1–6). This is the whole story (H1–H4) and a complete CS455 result.
- **Cut first if behind:** extra seeds (1–2) → then the rank 8-vs-16 ablation → then the sample-count finer sweep. Their absence costs only tight confidence intervals, not the conclusions.
- **Decision checkpoint = end of Day 5 (June 1).** If MVR isn't done, *stop adding scope*: spend remaining days on analysis + writing of what we have. A clean, honest single-seed result beats a half-finished multi-seed one.
- **Day 8 is the swing day:** on track → add seeds/ablation; behind → deepen analysis and start the report early.
- Days 9–10 (report + README + buffer) are **fixed and non-negotiable** — submission quality and reproducibility are graded; never borrow from them.

**Division of labor (suggested):**
- **Ali:** Stages 1–2 (generation, rejection sampling, dataset release) + efficiency/latency benchmarking.
- **Revna:** Stage 3 (QLoRA training, configs, seeds/ablation) + Stage 4 metrics/CIs.
- **Shared:** `prompts.py`/`answers.py` (must agree early), error analysis, report.

---

## 11. Risks & Contingencies (from proposal, with triggers)

| Risk | Trigger (measurable) | Plan B |
|---|---|---|
| **R1 — Teacher too weak** | Acceptance rate so low that condition iv lacks 3 traces for most problems | (a) raise k; (b) blend gold CoTs (hybrid mix); (c) swap teacher → `Qwen2.5-Math-7B-Instruct` (4-bit, fits T4) |
| **R2 — 1B too small** | Distilled 1B plateaus ≤ 40% GSM8K | Swap student → `Llama-3.2-3B-Instruct` (4-bit on L4); report 1B failure as a finding |
| **R3 — GSM8K contamination** | GSM8K gains but MATH-200 flat | Treat MATH-200 as primary genuine-generalization signal; flag caveat, constrain claims |
| **R4 — Colab time/disconnects** (new) | Runs out of GPU hours / session drops mid-experiment | Full operational strategy in [§9.1](#91-colab-operational-strategy): resumable shards + Drive checkpoints, env pinning, ≤2h runs, both teammates' accounts, **Kaggle (30 GPU-h/week)** as separate-quota fallback; de-scope to MVR per [§10 cut-lines](#10-compressed-timeline-may-28--june-7) |
| **R5 — Eval bugs** (new) | Suspiciously high/low acc | `tests/test_answers.py` on tricky cases; manually inspect 20 predictions per condition |

---

## 12. Final Deliverables (June 7)

1. **LoRA adapter** (best condition) → Hugging Face Hub.
2. **Cleaned teacher-CoT dataset** → Hugging Face Hub (derivative, documented).
3. **GitHub repo** (zip) → reproducible scripts + README that runs end-to-end on fresh Colab Pro.
4. **Final report** (~8–10 pages PDF): motivation, related work (KD + CoT distillation), method, results across all conditions, per-difficulty error analysis, MATH OOD, latency/cost, honest limitations (contamination), future work, **explicit AI-usage disclosure**.
5. **Demo** (after June 7, format TBA).

---

## 13. Immediate Next Steps (today, in order)

1. ✅ **Request gated access** to both Llama models on HF — *blocking, do first.*
2. Create the repo skeleton (`pyproject.toml`, `src/mathdistill/`, `configs/`, `notebooks/`, `tests/`).
3. Implement `prompts.py` + `answers.py` and their unit tests (this is the foundation everything depends on).
4. Build `00_setup.ipynb`: HF login, install, Drive mount, load both models in 4-bit, run one CoT generation each as a smoke test.

> Want me to scaffold the repository now — `pyproject.toml`, the `src/mathdistill/` modules (starting with `prompts.py` + `answers.py` + tests), the YAML configs, and the `00_setup` Colab notebook? Say the word and I'll generate the starter files.
