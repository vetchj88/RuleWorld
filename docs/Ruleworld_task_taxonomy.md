# RuleWorld-v1.0: A Benchmark for Compositional Continual Learning

### **The Core Philosophy**

Current Continual Learning (CL) benchmarks typically evaluate **Semantic Forgetting**—testing whether a model forgets facts or domain knowledge (e.g., forgetting Wikipedia data while learning Yelp reviews).

**RuleWorld-v1.0** tests **Algorithmic Forgetting and Compositional Resilience**. It evaluates a model's ability to learn distinct, programmatic functions (, ) sequentially and maintain those isolated rules when forced to combine them into complex, multi-step algorithmic hierarchies ().

---

### **Level 1: The Primitives (P-Series, Tasks P01–P20)**

The foundation of RuleWorld consists of 20 strictly defined algorithmic operations. These act as the fundamental "laws of physics" for the benchmark. They are divided into two distinct domains to test for representation isolation:

#### **Domain S: Symbol & String Manipulation (P01–P14)**

These tasks require spatial, structural, and matching manipulation over a 24-token vocabulary (`ba`, `be`, `bi`, etc.).

* **P01 (`REV`):** Reverse the sequence.
* **P02 (`ROTL`):** Rotate tokens left by .
* **P03 (`CHUNKREV`):** Reverse tokens in chunks of size .
* **P04 (`STRIDEKEEP`):** Keep tokens at stride  with offset .
* **P05 (`FILTERIN`):** Filter sequence to keep only tokens from a specified subset .
* **P06 (`SUBST`):** Substitute tokens based on a provided permutation list (`pi_list`).
* **P07 (`DUPLICATE`):** Duplicate specific tokens if they appear in subset .
* **P08 (`INSERTZZ`):** Insert the marker token `zz` after specific tokens in subset .
* **P09 (`PARTITION`):** Partition the sequence, moving tokens in subset  to the front.
* **P10 (`SORT`):** Sort the sequence lexicographically.
* **P11 (`UNIQUE`):** Remove duplicate tokens, keeping only the first occurrence.
* **P12 (`MIRROR`):** Reflect the input sequence (append the reversed sequence to the original).
* **P13 (`SLICE`):** Extract a specific subsequence starting at position  with width .
* **P14 (`RUNLEN`):** Perform run-length encoding. *(Note: This bridges domains, mapping )*.

#### **Domain D: Digit & Mathematical Operations (P15–P20)**

These tasks require arithmetic reasoning and state-tracking over a base-10 digit vocabulary (`d0`–`d9`).

* **P15 (`DREV`):** Reverse the sequence of digits.
* **P16 (`DROT`):** Rotate digits by .
* **P17 (`DADD`):** Add constant  to each digit (modulo 10).
* **P18 (`DMUL`):** Multiply each digit by  (modulo 10).
* **P19 (`DPREFIXSUM`):** Compute the prefix sum of the digits (modulo 10).
* **P20 (`DPERM`):** Permute the digits based on a specific 10-element mapping (`sigma_list`).

**Benchmark Challenge:** Standard Large Language Models struggle with precise, deterministic algorithmic manipulation. The P-Series establishes a baseline to ensure the model can achieve isolated mastery of pure rules before complexity scales.

---

### **Level 2: The Composites (C-Series, Tasks C01–C20)**

The C-Series evaluates the model's structural resilience by forcing it to execute multi-step reasoning. The model must chain previously learned primitives without destroying the isolated logic of those original rules.

#### **Domain S Composites: Structural Clashes (C01–C10)**

These composites combine string manipulation rules, testing the model's representational bottlenecks in spatial logic.

* **C01 (`ROTL`  `REV`):** Reverse, then rotate left ().
* **C02 (`SUBST`  `PARTITION`):** Partition by subset , then substitute via permutation list.
* **C03 (`FILTERIN`  `SORT`):** Sort lexicographically, then filter by subset .
* **C04 (`STRIDEKEEP`  `INSERTZZ`):** Insert `zz` markers, then keep stride ().
* **C05 (`CHUNKREV`  `MIRROR`):** Mirror the sequence, then reverse in chunks ().
* **C06 (`SLICE`  `DUPLICATE`):** Duplicate subset , then slice ().
* **C07 (`UNIQUE`  `PARTITION`):** Partition by subset , then remove duplicates.
* **C08 (`INSERTZZ`  `SUBST`):** Substitute via permutation, then insert `zz` markers.
* **C09 (`MIRROR`  `SLICE`):** Slice (), then mirror. *(A high-interference structural clash).*
* **C10 (`ROTL`  `FILTERIN`):** Filter by subset , then rotate left ().

#### **Domain D Composites: Arithmetic Clashes (C11–C20)**

These composites combine mathematical operations, forcing the model to retain operation order and modulo logic dynamically.

* **C11 (`DADD`  `DREV`):** Reverse digits, then add ().
* **C12 (`DREV`  `DPREFIXSUM`):** Compute prefix sum, then reverse.
* **C13 (`DPERM`  `DADD`):** Add (), then permute via mapping.
* **C14 (`DROT`  `DMUL`):** Multiply (), then rotate ().
* **C15 (`DPREFIXSUM`  `DPERM`):** Permute via mapping, then compute prefix sum.
* **C16 (`DMUL`  `DREV`):** Reverse, then multiply ().
* **C17 (`DADD`  `DROT`):** Rotate (), then add ().
* **C18 (`DPERM`  `DREV`):** Reverse, then permute via mapping.
* **C19 (`DPREFIXSUM`  `DADD`):** Add (), then compute prefix sum.
* **C20 (`DROT`  `DPREFIXSUM`):** Compute prefix sum, then rotate ().

---

### **Evaluation Metrics: What RuleWorld Measures**

To succeed on RuleWorld, an architecture must be tracked across a continuous  evaluation matrix to measure three specific phenomena:

1. **Catastrophic Algorithmic Forgetting:** Does learning the mathematical composite  overwrite the spatial weights that govern ?
2. **Domain Isolation:** Can the model successfully quarantine interference? (e.g., If the Symbol Composites  trigger a routing collapse, do the Digit Primitives  remain insulated and mathematically pristine?)
3. **Positive Backward Transfer (BWT+):** Does forcing the model to solve an algorithmic composite act as a structural regularizer that *increases* the zero-shot accuracy of its underlying primitive components?