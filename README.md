# LDPC Belief-Propagation Decoder

A reproducible Python project implementing and evaluating a **sum-product belief-propagation (BP) decoder for LDPC codes** over a binary symmetric channel (BSC).

## Why this project

This project is designed as a focused preparation project for research involving:

- error-correcting codes
- iterative decoding
- message-passing algorithms
- probabilistic inference
- machine-learning-assisted decoding

It complements a separate quantum-error-correction project by establishing the classical coding-theory and message-passing foundations.

> **Scope:** this repository implements classical LDPC decoding. It does not claim to implement a quantum LDPC decoder.

## Methods

The experiment includes:

1. A regular Gallager-style `(3, 6)` LDPC parity-check matrix.
2. Transmission of the all-zero codeword through a BSC.
3. A baseline hard-decision bit-flipping decoder.
4. A sum-product belief-propagation decoder using log-likelihood ratios (LLRs).
5. BER and block-error-rate evaluation over several channel crossover probabilities.
6. A plot comparing decoder performance.

## Project structure

```text
ldpc-belief-propagation-project/
├── README.md
├── requirements.txt
├── src/
│   ├── ldpc.py
│   └── run_experiment.py
├── tests/
│   └── test_ldpc.py
└── results/
    └── (generated plots and CSV files)
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the tests

```bash
python -m pytest
```

## Run the experiment

```bash
python src/run_experiment.py
```

The script creates:

```text
results/ber_vs_crossover_probability.png
results/ldpc_results.csv
```

The default experiment uses a moderate number of blocks so that it can run on a normal laptop. Increase `--blocks` for smoother curves.

Example:

```bash
python src/run_experiment.py --blocks 3000 --iterations 30 --seed 42
```

## Main algorithm

For a BSC with crossover probability `p`, the channel LLR for a received bit `y` is

```text
LLR(y) = log((1-p)/p)       if y = 0
         -log((1-p)/p)      if y = 1
```

The sum-product BP decoder passes messages between variable nodes and check nodes. Check-to-variable messages use the standard hyperbolic-tangent update:

```text
r_(c->v) = 2 atanh( product tanh(q_(v'->c)/2) )
```

The decoder stops early when the estimated codeword satisfies all parity checks.

## Reproducibility

The LDPC matrix generation and Monte Carlo experiment use explicit random seeds.

## Suggested research extensions

For a stronger follow-up project, consider:

- comparing BP with min-sum decoding;
- testing different LDPC block lengths;
- studying decoder performance versus maximum iterations;
- adding AWGN channels;
- testing irregular LDPC matrices;
- learning-based correction of BP messages;
- comparing classical LDPC decoding with a small quantum-error-correction decoding experiment.

## Suggested CV wording

**LDPC Belief-Propagation Decoder | Python | NumPy | SciPy | 2026**

- Implemented a sum-product belief-propagation decoder for a regular Gallager-style LDPC code over a binary symmetric channel.
- Generated sparse parity-check matrices and performed iterative variable-to-check and check-to-variable message updates using log-likelihood ratios.
- Compared probabilistic BP decoding with a hard-decision bit-flipping baseline across multiple channel-noise levels.
- Evaluated bit-error rate, block-error rate, convergence, and parity-check satisfaction using reproducible Monte Carlo experiments.
