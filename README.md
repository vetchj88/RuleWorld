# RuleWorld
RuleWorld is a Compositional Continual Learning benchmark testing algorithmic resilience, not just semantic facts. Featuring 40 procedurally generated tasks, it measures a model's ability to quarantine interference (Domain Isolation) and achieve Positive Backward Transfer (BWT+) when chaining primitives into complex multi-step composites.



# 🌍 RuleWorld: A Benchmark for Compositional Continual Learning

Most Continual Learning (CL) benchmarks test **Semantic Forgetting**—asking if a Large Language Model forgets facts from Wikipedia while fine-tuning on medical journals.

**RuleWorld** tests **Algorithmic Forgetting and Compositional Resilience**. It evaluates whether a model can sequentially learn distinct, deterministic programmatic functions (, ) and maintain those isolated rules when pressured to combine them into complex algorithmic hierarchies ().

Standard LLMs and traditional CL methods (like Sequential LoRA) fail spectacularly on RuleWorld due to representational collapse and catastrophic algorithmic forgetting.

---

## 🧩 The Benchmark Taxonomy

RuleWorld consists of a curriculum of **40 tasks**, generated procedurally to prevent test-set contamination. The tasks are split into two phases and two orthogonal domains: **Symbol/String (S)** and **Digit/Math (D)**.

### Phase 1: The Primitives (P01 – P20)

20 single-step algorithmic operations that establish the fundamental "laws of physics" for the benchmark.

* **Domain S (String Manipulation):** Tasks requiring spatial and structural token manipulation over a 24-token vocabulary. Examples: `ROTL` (Rotate Left), `MIRROR`, `SLICE`, `CHUNKREV`.
* **Domain D (Arithmetic Operations):** Tasks requiring state-tracking over base-10 digits. Examples: `DMUL` (Digit Multiply), `DADD` (Digit Add), `DPREFIXSUM`.

### Phase 2: The Composites (C01 – C20)

20 multi-step reasoning tasks that force the model to chain the Primitives. This tests if the model's architecture can route and compose logic without destroying the underlying rules.

* **Intra-Domain Clashes:** E.g., `C09 (MIRROR ∘ SLICE)` — Extract a substring and mirror it. Known to cause severe router mode collapse in dynamic networks.
* **Inter-Domain Clashes:** Forcing the model to maintain strict operational order across mathematical and structural boundaries.

📖 Deep Dive: For a comprehensive mathematical breakdown of all 20 Primitives and 20 Composites, including vocabulary constraints and permutation logic, read the full RuleWorld Task Taxonomy.

---

## 📊 Evaluation Metrics

To succeed on RuleWorld, an architecture must be evaluated across a continuous  matrix. We measure three specific phenomena:

1. **Catastrophic Algorithmic Forgetting:** Does learning the mathematical composite `C14` overwrite the spatial weights that govern primitive `P01`?
2. **Domain Isolation:** Can the architecture quarantine interference? If String Composites cause an interference event, do the Math Primitives remain insulated and mathematically pristine?
3. **Positive Backward Transfer (BWT+):** The ultimate test of compositional generalization. Does forcing the model to solve a composite task act as a structural regularizer that *increases* the zero-shot accuracy of its underlying primitive components?

---

## 🚀 Provided Baselines

This repository includes the data generation scripts, evaluation harness, and code for three standard open-source baselines to demonstrate the difficulty of the benchmark:

1. **Frozen Zero-Shot (`baselines/baseline_eval.py`):** Standard inference on base LLMs (e.g., Llama-3, Mistral) without training. Proves that models cannot natively execute algorithmic compositions reliably.
2. **Sequential LoRA (`baselines/baseline_seq_lora.py`):** The naive Continual Learning approach. Training a single LoRA adapter sequentially from Task 1 to Task 40. Exhibits massive, undeniable catastrophic forgetting by Task C05.
3. **Best-of-N (`baselines/baseline_bon.py`):** Inference-time scaling (generating *N* responses and selecting the most consistent). Proves that compute scaling alone cannot resolve algorithmic structural clashes.

---

## 🏆 The Reference Architecture: MorphoLayer (Proprietary)

While standard architectures fail RuleWorld, it is solvable.

Our proprietary reference architecture, **MorphoLayer** (a dynamic adapter-spawning network), currently holds the State-of-the-Art (SOTA) on RuleWorld. It demonstrates:

* **Domain Isolation:** Successfully partitioning the "Digit Domain" weights from the "Symbol Domain", preventing cross-domain catastrophic collapse during high-interference tasks.
* **11x Positive Backward Transfer (BWT+):** LevinLayer achieved up to an 11x performance multiplier on underlying primitives (`P12`, `P13`) *after* being trained on their composite structures, proving that algorithmic composition can act as a performance enhancer in properly routed networks.

*Can your architecture beat the LevinLayer baselines? We invite the community to try.*

---

## 🛠️ Quick Start

### 1. Generate the Dataset

RuleWorld data is generated procedurally via the provided manifest.

```bash
git clone https://github.com/YourUsername/RuleWorld.git
cd RuleWorld
python scripts/generate_data.py --manifest manifests/ruleworld_manifest_v1.json --samples_per_task 5000 --seed 20260126

```

### 2. Run a Baseline

Evaluate the Sequential LoRA baseline to observe catastrophic forgetting:

```bash
python baselines/baseline_seq_lora.py --data_root data/ruleworld_v1 --model "meta-llama/Meta-Llama-3-8B"

```

### 3. Evaluate Your Own Model

Use the standard evaluation harness to generate your  matrix:

```bash
python scripts/evaluation.py --predictions my_model_outputs.jsonl --ground_truth data/ruleworld_v1/test/

```

---

## 📄 License & Citation

RuleWorld is released under the [MIT License](./LICENSE).

If you use RuleWorld in your research, please cite:

```bibtex
@misc{Vetch2026ruleworld,
  author = {Vetch, Justin M.},
  title = {RuleWorld: A Benchmark for Compositional Continual Learning},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/vetchj88/RuleWorld}}
}
