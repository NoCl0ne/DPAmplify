# DPAmplify

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status: Research in Progress](https://img.shields.io/badge/status-research%20in%20progress-orange.svg)]()
[![arXiv: forthcoming](https://img.shields.io/badge/arXiv-forthcoming-red.svg)]()

## DPAmplify: Noise-Aware Byzantine Attacks Exploiting the Analytical Structure of the Gaussian DP Mechanism in Federated Learning

---

### Overview

DPAmplify is a research project that identifies and demonstrates a previously 
uncharacterized attack vector in differentially-private federated learning (DP-FL).

The common assumption in the federated learning security literature is that 
differential privacy (DP) noise reduces the effectiveness of Byzantine attacks 
by masking malicious gradient updates. **DPAmplify inverts this assumption.**

We show that a Byzantine participant who knows (or can passively estimate) the 
clipping threshold C and noise standard deviation σ of the Gaussian DP mechanism 
can construct adversarial gradients whose **expected post-noise value converges 
coherently toward an adversarial target direction** — achieving a formal 
signal-to-noise ratio (SNR) advantage over honest participants, even under 
Byzantine-robust aggregation rules such as Krum and TrimmedMean.

This work challenges the widely held belief that combining DP with 
Byzantine-robust aggregation provides defense-in-depth in federated learning.

---

### Key Idea

The Gaussian DP mechanism applied to a gradient `g` is:
```
M_DP(g) = clip(g, C) + ξ,    ξ ~ N(0, σ²I)
```

Since `E[ξ] = 0`, the expected output satisfies:
```
E[M_DP(g)] = clip(g, C)
```

A Byzantine participant sets `g_adv = C · (g_target / ‖g_target‖)`.  
Because `‖g_adv‖ = C`, the clipping operation is a no-op, and:
```
E[M_DP(g_adv)] = g_adv  →  aligned with g_target
```

With `k` Byzantine clients repeating this over `T` rounds, the adversarial 
signal accumulates coherently while honest gradients undergo random-walk 
cancellation, yielding a formal SNR advantage of:
```
SNR_attack = k · ‖g_target‖ / (σ · √(n - k))
```

---

### Differentiation from Prior Work

| Work | What it does | Why DPAmplify is different |
|------|-------------|---------------------------|
| **Robust-HDP** (Malekmohammadi et al., ICML 2024) | Noise-aware *aggregation* — server estimates DP noise levels to improve utility | Defensive, server-side. DPAmplify is an *offensive* client-side attack that exploits the same noise structure |
| **LIE** (Baruch et al., NeurIPS 2019) | Crafts malicious gradients using statistics of honest updates to evade detection | Does not consider the DP mechanism. DPAmplify specifically exploits the analytical structure of the Gaussian mechanism |
| **MinMax / MinSum** (Shejwalkar & Houmansadr, USENIX 2021) | Optimizes perturbation to evade aggregator metrics | No DP-awareness. DPAmplify uses `E[M_DP(·)]` as the explicit optimization objective |
| **FLTrust** (Cao et al., NDSS 2022) | Uses a server-side root dataset for gradient scoring | Vulnerable when Byzantine clients can passively estimate the root gradient direction |

**Key novelty confirmed by literature search (March 2026):** no existing paper 
uses `E[M_DP(g_adv)]` as the optimization objective for a Byzantine attack, 
nor formally proves the SNR advantage `k/√(n-k)` in DP-FL settings.

---

### Threat Model

- **Setting:** Cross-silo federated learning without Secure Aggregation (SecAgg)
- **Attacker:** `k` Byzantine clients (k < n/2) under honest-but-curious server
- **Attacker knowledge:** Black-box access to the FL system; can observe 
  aggregated gradient norms across rounds
- **Attacker capability:** Controls gradient updates submitted by k clients; 
  cannot modify server-side aggregation or other clients' updates
- **Attack phases:**
  - **Phase 1 (Passive estimation):** Byzantine clients behave honestly while 
    observing aggregated gradient norms to estimate C and σ without detection
  - **Phase 2 (Attack):** Byzantine clients submit `g_adv` optimized to 
    accumulate toward `g_target` in expectation

**Out of scope:** Settings with SecAgg, trusted execution environments, 
or cryptographic gradient verification. These are documented as 
explicit limitations in the paper.

---

### Repository Structure
```
dpamplify/
│
├── theory/                        # Formal mathematical foundations
│   ├── dp_mechanism.py            # Gaussian DP mechanism implementation
│   ├── snr_analysis.py            # Theoretical SNR computation
│   └── proofs/
│       └── theorem_snr.tex        # Formal LaTeX proof of SNR theorem
│
├── attack/                        # Attack implementation
│   ├── parameter_estimator.py     # Passive estimation of C and σ
│   ├── gradient_optimizer.py      # Computation of optimal g_adv
│   └── byzantine_client.py        # DPAmplify Byzantine client (Flower)
│
├── fl_system/                     # Federated learning infrastructure
│   ├── server.py                  # FL server with DP and logging
│   ├── honest_client.py           # Honest client with Opacus DP-SGD
│   └── aggregators/
│       ├── fedavg.py              # FedAvg baseline
│       ├── krum.py                # Krum (Blanchard et al. 2017)
│       └── trimmed_mean.py        # TrimmedMean (Yin et al. 2018)
│
├── experiments/                   # Reproducible experiments
│   ├── exp_01_snr_validation.py   # Empirical SNR vs theoretical bound
│   ├── exp_02_mnist_attack.py     # Main attack on MNIST
│   ├── exp_03_evasion.py          # Evasion of Krum and TrimmedMean
│   └── exp_04_adaptive_clipping.py# Robustness under adaptive clipping
│
├── countermeasures/               # Proposed defenses
│   ├── randomized_clipping.py     # Randomized clipping threshold
│   └── gradient_auditor.py        # Statistical auditing of gradients
│
├── tests/                         # Unit and integration tests
│   └── test_dp_mechanism.py
│
├── notebooks/
│   └── demo.ipynb                 # Interactive demonstration
│
├── paper/
│   ├── dpamplify_arxiv.tex        # arXiv preprint (IEEE format)
│   └── figures/                   # Generated figure scripts
│
├── requirements.txt
├── setup.py
├── .gitignore
├── CONTRIBUTING.md
└── README.md
```

---

### Installation

**Requirements:** Python 3.11, pip
```bash
# Clone the repository
git clone https://github.com/[YOUR_USERNAME]/dpamplify.git
cd dpamplify

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

**Verify installation:**
```bash
python -c "from theory.dp_mechanism import DPMechanism; print('OK')"
pytest tests/ -v
```

---

### Quick Start

**1. Verify the theoretical SNR bounds empirically:**
```bash
python experiments/exp_01_snr_validation.py
```

This generates `paper/figures/fig_snr_validation.pdf` comparing the 
empirical SNR against the theoretical bound across values of σ and k.

**2. Run the main attack on MNIST:**
```bash
python experiments/exp_02_mnist_attack.py \
    --n_clients 20 \
    --n_byzantine 3 \
    --clipping_C 1.0 \
    --noise_sigma 0.1 \
    --n_rounds 100 \
    --aggregator fedavg
```

**3. Test evasion of robust aggregators:**
```bash
python experiments/exp_03_evasion.py --aggregator krum
python experiments/exp_03_evasion.py --aggregator trimmed_mean
```

**4. Interactive demo:**
```bash
jupyter notebook notebooks/demo.ipynb
```

---

### Experimental Configuration (Baseline)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total clients n | 20 | — |
| Byzantine clients k | 3 | 15% of total |
| Clipping threshold C | 1.0 | Initial value; estimated passively |
| Noise multiplier σ | 0.1 | Varied in ablation |
| Estimation rounds T_est | 20 | Passive phase |
| Attack rounds T_attack | 100 | Active phase |
| Model | MLP 784→128→10 | MNIST classification |
| Local epochs | 2 | Per round |
| Batch size | 64 | — |
| Aggregator (baseline) | FedAvg | Also tested: Krum, TrimmedMean |

---

### Scope and Limitations

This research applies to federated learning deployments that:

- Use the Gaussian DP mechanism (DP-SGD with L2 clipping and Gaussian noise)
- Do **not** use Secure Aggregation (SecAgg)
- Do **not** use cryptographic gradient verification (e.g., ZK-proofs)
- Have a static or slowly adaptive clipping threshold

The attack does **not** apply to settings protected by SecAgg, trusted 
hardware enclaves, or the GradAttest protocol. These are documented 
explicitly as out-of-scope in Section 6 of the paper.

---

### Ethical Statement

This research is conducted to expose a structural vulnerability in 
differentially-private federated learning and motivate the development 
of stronger defenses. The attack is implemented and tested exclusively 
in isolated simulation environments. No real-world federated learning 
systems were targeted during this research.

The paper includes a **Countermeasures** section of equal prominence to 
the attack description. Responsible disclosure to maintainers of Flower 
and Opacus will occur prior to public release of the full attack code.

The full attack implementation (`attack/byzantine_client.py`) will be 
released 90 days after paper publication to allow framework maintainers 
to implement mitigations.

---

### Paper

*Preprint forthcoming on arXiv.*
```bibtex
@misc{dpamplify2026,
  title     = {DPAmplify: Noise-Aware Byzantine Attacks Exploiting 
               the Analytical Structure of the Gaussian DP Mechanism 
               in Federated Learning},
  author    = {[Author]},
  year      = {2026},
  note      = {Preprint. Work in progress.
               \url{https://github.com/[YOUR_USERNAME]/dpamplify}}
}
```

---

### License

Copyright 2026 [Author]

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) 
for the full license text.

You may use this software for research purposes with attribution. 
Commercial use is permitted under the terms of Apache 2.0.
```

---

## PROMPT COMPLETO PER L'AGENT
```
You are a senior software engineer and research assistant. Your task is
to generate the complete file and folder structure for the DPAmplify
research repository. Create every file listed below with real, working,
complete content — no placeholder comments like "# TODO implement this"
in Python files. LaTeX files may use [TODO] markers only inside paper
section bodies, not in formal definitions.

All Python code must be:
- PEP 8 compliant
- Fully type-annotated
- Documented with complete docstrings (Args, Returns, Raises)
- Immediately runnable without modification

══════════════════════════════════════════════════════════
PROJECT CONTEXT
══════════════════════════════════════════════════════════

Project name: DPAmplify
Language: Python 3.11
License: Apache 2.0
Topic: Byzantine attack on differentially-private federated learning

CORE MATHEMATICAL IDEA:
The Gaussian DP mechanism applied to gradient g is:
    M_DP(g) = clip(g, C) + ξ,  where ξ ~ N(0, σ²I)
    clip(g, C) = g · min(1, C / ‖g‖₂)

Since E[ξ] = 0:
    E[M_DP(g)] = clip(g, C)

A Byzantine client sets:
    g_adv = C · (g_target / ‖g_target‖₂)

Because ‖g_adv‖₂ = C exactly, clip(g_adv, C) = g_adv, so:
    E[M_DP(g_adv)] = g_adv  (aligned with g_target direction)

With k Byzantine clients and n total clients, the theoretical
SNR advantage of the attack is:
    SNR_attack = k · ‖g_target‖ / (σ · √(n - k))

The attack has two phases:
- Phase 1 (passive estimation, rounds 1..T_est):
    Byzantine client behaves honestly but observes
    aggregated gradient norms to estimate C and σ.
    Estimation: C ≈ percentile_90(observed_norms)
                σ ≈ std(observed_norms - mean(observed_norms))

- Phase 2 (attack, rounds T_est+1..T_end):
    Byzantine client sends g_adv = C · (g_target / ‖g_target‖₂)
    regardless of local data.

DEPENDENCIES:
flwr>=1.8.0, torch>=2.2.0, torchvision>=0.17.0,
opacus>=1.4.0, scipy>=1.11.0, numpy>=1.26.0,
matplotlib>=3.8.0, seaborn>=0.13.0, pytest>=7.4.0,
jupyter>=1.0.0, tqdm>=4.66.0, pandas>=2.1.0

══════════════════════════════════════════════════════════
FILES TO CREATE
══════════════════════════════════════════════════════════

────────────────────────────────────────
FILE: requirements.txt
────────────────────────────────────────
List all dependencies above with pinned minimum versions.
One package per line, format: package>=version

────────────────────────────────────────
FILE: setup.py
────────────────────────────────────────
Standard setuptools setup.py with:
- name="dpamplify"
- version="0.1.0"
- python_requires=">=3.11"
- install_requires read from requirements.txt
- author="[Author]"
- description="Byzantine attacks exploiting the Gaussian DP 
  mechanism in federated learning"
- url="https://github.com/[YOUR_USERNAME]/dpamplify"

────────────────────────────────────────
FILE: .gitignore
────────────────────────────────────────
Comprehensive .gitignore for:
- Python (__pycache__, *.pyc, .venv, dist, build, *.egg-info)
- PyTorch (*.pt, *.pth, *.ckpt, checkpoints/, saved_models/)
- Jupyter (.ipynb_checkpoints)
- Datasets (data/, datasets/, *.csv, *.gz, *.zip) 
  EXCEPTION: keep tests/fixtures/*.csv
- Experiment outputs (results/, outputs/, logs/, *.log, *.json)
  EXCEPTION: keep paper/figures/*.json
- LaTeX (*.aux, *.bbl, *.blg, *.fdb_latexmk, *.fls, *.log, 
         *.out, *.synctex.gz, *.toc)
  EXCEPTION: keep paper/dpamplify_final.pdf
- IDE (.idea/, .vscode/, *.swp)
- OS (.DS_Store, Thumbs.db)
- Secrets (.env, secrets.py, *.key, api_keys.txt) — with comment
  "NEVER commit credentials"

────────────────────────────────────────
FILE: CONTRIBUTING.md
────────────────────────────────────────
Short contribution guide:
- This is an active research project
- Bug reports via GitHub Issues
- Pull requests welcome after paper publication
- Ethical note: code is for defensive security research only

────────────────────────────────────────
FILE: theory/__init__.py
────────────────────────────────────────
Module docstring: "Theoretical foundations of DPAmplify:
formal model of the Gaussian DP mechanism and SNR analysis."
Import DPMechanism and the SNR functions for convenience.

────────────────────────────────────────
FILE: theory/dp_mechanism.py
────────────────────────────────────────
Module docstring explaining the Gaussian DP mechanism.

Class: DPMechanism
  Purpose: Implements M_DP(g) = clip(g, C) + N(0, σ²I)

  __init__(self, C: float, sigma: float):
    Args:
      C: L2 clipping threshold (must be > 0)
      sigma: Gaussian noise standard deviation (must be > 0)
    Raises: ValueError if C <= 0 or sigma <= 0

  clip(self, g: np.ndarray) -> np.ndarray:
    Clips gradient to L2 norm C.
    Formula: g * min(1.0, C / max(np.linalg.norm(g), 1e-10))
    Returns gradient with ‖result‖₂ ≤ C.
    Docstring must include the formula.

  add_noise(self, g: np.ndarray) -> np.ndarray:
    Adds Gaussian noise N(0, sigma²I) to gradient.
    Returns g + noise where noise ~ N(0, sigma²) iid per coordinate.

  apply(self, g: np.ndarray) -> np.ndarray:
    Applies full DP mechanism: clip then add noise.
    Returns clip(g, C) + N(0, sigma²I).

  expected_output(self, g: np.ndarray) -> np.ndarray:
    Returns the expected value E[M_DP(g)] = clip(g, C).
    Explanation in docstring: noise has zero mean, so expectation
    equals the clipped gradient.

  sample_outputs(self, g: np.ndarray, n_samples: int) -> np.ndarray:
    Draws n_samples independent realizations of M_DP(g).
    Returns array of shape (n_samples, len(g)).

At module level, add a simple __main__ block that:
- Creates DPMechanism(C=1.0, sigma=0.1)
- Creates a test gradient g = [2.0, 0.0, 0.0] (norm > C, will be clipped)
- Prints clip(g), expected_output(g), and the mean of 1000 samples
- Verifies empirically that sample mean ≈ expected_output

────────────────────────────────────────
FILE: theory/snr_analysis.py
────────────────────────────────────────
Module docstring: "SNR analysis for the DPAmplify attack.
All functions implement the theoretical bounds derived in
theorem_snr.tex."

Function: compute_attack_snr(k: int, n: int, 
                              g_target_norm: float,
                              sigma: float) -> float:
  Computes theoretical SNR_attack = k * g_target_norm / (sigma * sqrt(n-k))
  Args:
    k: number of Byzantine clients
    n: total number of clients
    g_target_norm: L2 norm of the target gradient direction (float > 0)
    sigma: DP noise standard deviation
  Returns: float — theoretical SNR advantage of the attack
  Raises: ValueError if k >= n
  Docstring must state the formula explicitly.

Function: compute_honest_snr(n: int, k: int,
                              mu_honest_norm: float,
                              sigma: float) -> float:
  Computes SNR for honest gradient signal after aggregation.
  Formula: (n-k) * mu_honest_norm / (sigma * sqrt(n-k))
         = mu_honest_norm * sqrt(n-k) / sigma
  Args: n total clients, k Byzantine, mu_honest_norm = ‖E[g_honest]‖₂
  Returns: float

Function: snr_ratio(k: int, n: int,
                    g_target_norm: float,
                    mu_honest_norm: float) -> float:
  Computes SNR_attack / SNR_honest.
  This ratio is sigma-independent.
  Returns: k * g_target_norm / (mu_honest_norm * sqrt(n-k))

Function: plot_snr_vs_parameters(n: int = 20,
                                  sigma: float = 0.1,
                                  save_path: str = None):
  Generates a 2x2 matplotlib figure:
  - Top left: SNR_attack vs k (k from 1 to n//2)
  - Top right: SNR_attack vs sigma (sigma from 0.01 to 1.0)
  - Bottom left: SNR ratio vs k for different mu_honest_norm values
  - Bottom right: Minimum k needed to achieve SNR_attack > 1
  Uses seaborn style. Saves to save_path if provided.

────────────────────────────────────────
FILE: theory/proofs/theorem_snr.tex
────────────────────────────────────────
Standalone LaTeX file (not part of the main paper) containing
the formal proof of the SNR theorem. Content:

\documentclass{article}
\usepackage{amsmath, amssymb, amsthm}

Environments: \newtheorem{theorem}{Theorem}
              \newtheorem{lemma}{Lemma}
              \newtheorem{corollary}{Corollary}

DEFINITION 1: Gaussian DP Mechanism
  M_DP: R^d → R^d
  M_DP(g) = clip(g, C) + ξ,  ξ ~ N(0, σ²I_d)
  clip(g, C) = g · min(1, C/‖g‖₂)

LEMMA 1: (Zero-mean noise)
  For any fixed g ∈ R^d: E[M_DP(g)] = clip(g, C)
  Proof: E[ξ] = 0 component-wise. QED.

LEMMA 2: (Clipping-free adversarial gradient)
  If g_adv = C · (g_target / ‖g_target‖₂), then ‖g_adv‖₂ = C,
  therefore clip(g_adv, C) = g_adv.
  Proof: ‖g_adv‖₂ = C · ‖g_target / ‖g_target‖₂‖₂ = C · 1 = C.
  Since ‖g_adv‖₂ = C ≤ C, min(1, C/‖g_adv‖₂) = 1. QED.

THEOREM 1: (SNR Advantage of DPAmplify)
  Consider a DP-FL system with n clients, k Byzantine clients
  using DPAmplify, Gaussian mechanism with parameters (C, σ).
  Let g_target ∈ R^d be the adversarial target direction.
  Let μ_honest = E[g_honest] be the expected honest gradient.
  
  After T rounds of aggregation (FedAvg), the expected 
  contribution of Byzantine clients to the global update 
  is aligned with g_target with SNR:
  
    SNR_attack = (k · C) / (σ · √(n - k))   [per unit of ‖g_target‖/C]
  
  While the honest signal SNR is:
  
    SNR_honest = (‖μ_honest‖ · √(n-k)) / σ
  
  Proof: Write out the aggregated gradient as sum of Byzantine 
  and honest contributions. Apply Lemma 1 and Lemma 2 for the
  Byzantine term. Apply CLT for the honest term. Compute ratio.

COROLLARY 1: The SNR advantage is σ-independent when normalized.
  SNR_attack / SNR_honest = k·C / (‖μ_honest‖ · (n-k))

────────────────────────────────────────
FILE: attack/__init__.py
────────────────────────────────────────
Module docstring: "DPAmplify attack implementation.
Two-phase attack: passive parameter estimation followed by
optimized adversarial gradient submission."

────────────────────────────────────────
FILE: attack/parameter_estimator.py
────────────────────────────────────────
Module docstring: "Passive estimation of DP mechanism parameters
(clipping threshold C and noise standard deviation σ) from
observed aggregated gradient norms. The Byzantine client behaves
as an honest client during this phase and is not detectable."

Class: PassiveParameterEstimator
  __init__(self, history_window: int = 20,
           percentile_C: float = 90.0):
    Args:
      history_window: number of rounds to collect before estimating
      percentile_C: percentile of observed norms used to estimate C
    Initialize: self._norm_history = []

  update(self, observed_norm: float) -> None:
    Appends observed aggregated gradient norm to history.
    Called once per FL round during passive phase.

  estimate_C(self) -> float:
    Returns np.percentile(self._norm_history, self.percentile_C)
    Raises RuntimeError if not is_ready()

  estimate_sigma(self) -> float:
    Estimates noise std as standard deviation of observed norms
    minus the running mean.
    Formula: np.std(self._norm_history)
    Note: this is an approximation; true sigma estimation is 
    discussed in Section 4 of the paper.
    Raises RuntimeError if not is_ready()

  is_ready(self) -> bool:
    Returns True if len(self._norm_history) >= self.history_window

  get_estimates(self) -> dict:
    Returns {"C": self.estimate_C(), "sigma": self.estimate_sigma(),
             "n_observations": len(self._norm_history)}
    Raises RuntimeError if not is_ready()

  At module level __main__: simulate 25 observed norms drawn from
  a distribution consistent with clipping C=1.0 and sigma=0.1,
  run the estimator, and print accuracy of estimates.

────────────────────────────────────────
FILE: attack/gradient_optimizer.py
────────────────────────────────────────
Module docstring: "Computation of the optimal adversarial gradient
g_adv = C · (g_target / ‖g_target‖₂) such that the expected
post-DP-mechanism output E[M_DP(g_adv)] is aligned with g_target."

Class: GradientOptimizer
  __init__(self, g_target: np.ndarray, C: float):
    Args:
      g_target: target adversarial direction (any nonzero vector)
      C: estimated clipping threshold
    Internally normalizes g_target and stores as unit vector.

  compute_g_adv(self) -> np.ndarray:
    Returns g_adv = C * (g_target / ‖g_target‖₂)
    Docstring explains: ‖g_adv‖₂ = C exactly, so clip(g_adv, C) = g_adv,
    and therefore E[M_DP(g_adv)] = g_adv = C * g_target_unit.

  compute_expected_contribution(self, k: int, n: int) -> np.ndarray:
    Returns (k / n) * self.compute_g_adv()
    This is the expected Byzantine contribution to FedAvg per round.

  verify_no_clipping(self, mechanism: "DPMechanism") -> bool:
    Verifies empirically (1000 samples) that the mean of
    mechanism.apply(self.compute_g_adv()) ≈ self.compute_g_adv().
    Returns True if L2 error < 0.01 * C.

────────────────────────────────────────
FILE: attack/byzantine_client.py
────────────────────────────────────────
Module docstring: "DPAmplify Byzantine client implementation for
the Flower federated learning framework. Implements a two-phase
attack: passive estimation of DP parameters, then adversarial
gradient submission."

NOTE: This file implements the core attack. Per the project's
ethical policy, it will be released 90 days after paper
publication. The implementation is functional and complete
for research purposes.

Class: DPAmplifyClient(flwr.client.NumPyClient)
  __init__(self,
           cid: str,
           model: torch.nn.Module,
           train_loader: torch.utils.data.DataLoader,
           g_target: np.ndarray,
           estimator: PassiveParameterEstimator,
           T_est: int = 20,
           device: str = "cpu"):
    Args:
      cid: client ID string
      model: PyTorch model (used honestly during estimation phase)
      train_loader: local dataset loader
      g_target: target adversarial direction
      estimator: PassiveParameterEstimator instance
      T_est: number of estimation rounds before switching to attack
      device: torch device string

  _current_round(self) -> int:
    Returns internal round counter (incremented in fit()).

  _is_attack_phase(self) -> bool:
    Returns True if _current_round() > T_est and estimator.is_ready()

  get_parameters(self, config: dict) -> List[np.ndarray]:
    Returns current model parameters as list of numpy arrays.

  fit(self, parameters: List[np.ndarray],
      config: dict) -> Tuple[List[np.ndarray], int, dict]:
    If estimation phase:
      - Load parameters into model
      - Train one epoch normally (honest behavior)
      - Compute gradient norms and update estimator
      - Return honest gradient update
      - Log: f"[Byzantine {self.cid}] Round {r}: ESTIMATION phase"
    If attack phase:
      - Create optimizer with estimator.get_estimates()
      - Compute g_adv for every parameter tensor
      - Return g_adv as the parameter update
      - Log: f"[Byzantine {self.cid}] Round {r}: ATTACK phase"
    Always increment round counter.
    Returns (parameters, n_samples, metrics_dict)

  evaluate(self, parameters, config):
    Standard evaluation — Byzantine behaves honestly here
    to avoid detection via evaluation metrics.
    Returns (loss, n_samples, {"accuracy": acc})

────────────────────────────────────────
FILE: fl_system/__init__.py
────────────────────────────────────────
Module docstring: "Federated learning infrastructure for DPAmplify
experiments. Provides server, honest client, and aggregator
implementations."

────────────────────────────────────────
FILE: fl_system/server.py
────────────────────────────────────────
Module docstring: "FL server with DP-compatible aggregation and
gradient norm logging (required for Byzantine passive estimation
experiments)."

Function: create_strategy(aggregator_name: str,
                           min_clients: int,
                           fraction_fit: float = 1.0
                           ) -> flwr.server.strategy.Strategy:
  Creates and returns the appropriate Flower strategy.
  Supports: "fedavg", "krum", "trimmed_mean"
  For non-standard aggregators, wraps FedAvg and overrides
  aggregate_fit with the custom aggregator logic.
  Raises ValueError for unknown aggregator names.

Function: run_server(aggregator_name: str = "fedavg",
                     n_rounds: int = 100,
                     min_clients: int = 2,
                     server_address: str = "[::]:8080") -> None:
  Starts a Flower server. Logs aggregated update norms to
  "logs/gradient_norms.csv" for analysis.

────────────────────────────────────────
FILE: fl_system/honest_client.py
────────────────────────────────────────
Module docstring: "Honest FL client using Opacus for DP-SGD.
Implements standard federated learning with differential privacy."

Class: HonestClient(flwr.client.NumPyClient)
  __init__(self,
           cid: str,
           model: torch.nn.Module,
           train_loader: torch.utils.data.DataLoader,
           test_loader: torch.utils.data.DataLoader,
           C: float = 1.0,
           sigma: float = 0.1,
           local_epochs: int = 2,
           device: str = "cpu"):

  get_parameters(self, config) -> List[np.ndarray]:
    Returns model parameters.

  fit(self, parameters, config) -> Tuple[...]:
    Loads parameters, trains for local_epochs using Opacus
    DP-SGD with clipping C and noise sigma. Returns updated
    parameters, dataset size, and empty metrics dict.
    
    Opacus setup:
      optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
      privacy_engine = opacus.PrivacyEngine()
      model, optimizer, train_loader = privacy_engine.make_private(
          module=model,
          optimizer=optimizer,
          data_loader=train_loader,
          noise_multiplier=sigma,
          max_grad_norm=C)

  evaluate(self, parameters, config) -> Tuple[...]:
    Evaluates model on test_loader.
    Returns (loss, dataset_size, {"accuracy": accuracy})

────────────────────────────────────────
FILE: fl_system/aggregators/fedavg.py
────────────────────────────────────────
Function: fedavg_aggregate(gradient_list: List[np.ndarray],
                            weights: List[int] = None) -> np.ndarray:
  Weighted average of gradients.
  If weights=None, uses equal weights.
  Returns element-wise weighted mean as np.ndarray.

────────────────────────────────────────
FILE: fl_system/aggregators/krum.py
────────────────────────────────────────
Module docstring includes citation:
  Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J.
  "Machine Learning with Adversaries: Byzantine Tolerant Gradient
  Descent." NeurIPS 2017.

Function: krum_aggregate(gradients: List[np.ndarray],
                          f: int) -> np.ndarray:
  Selects the gradient with minimum sum of distances to its
  (n - f - 2) nearest neighbors, where n = len(gradients).

  Args:
    gradients: list of gradient vectors (all same shape)
    f: number of assumed Byzantine workers
  Returns: the selected gradient (single vector, not averaged)
  Raises: ValueError if f >= len(gradients) // 2

  Algorithm:
    1. Compute pairwise L2 distances between all gradients
    2. For each gradient i, sort distances to all other gradients
    3. Sum the (n - f - 2) smallest distances for gradient i
    4. Return gradient with minimum such sum

  Include a worked example in the docstring with n=5, f=1.

────────────────────────────────────────
FILE: fl_system/aggregators/trimmed_mean.py
────────────────────────────────────────
Module docstring includes citation:
  Yin, D., Chen, Y., Kannan, R., & Bartlett, P.
  "Byzantine-robust distributed learning: Towards optimal
  statistical rates." ICML 2018.

Function: trimmed_mean_aggregate(gradients: List[np.ndarray],
                                   beta: float = 0.1) -> np.ndarray:
  Coordinate-wise trimmed mean.

  Args:
    gradients: list of gradient vectors (all same shape)
    beta: fraction to trim from each end (0 < beta < 0.5)
  Returns: np.ndarray of same shape as input gradients

  Algorithm:
    For each coordinate d:
      - Collect values[d] = [g[d] for g in gradients]
      - Sort values[d]
      - Remove floor(beta * n) values from each end
      - Compute mean of remaining values

  Raises: ValueError if beta <= 0 or beta >= 0.5

────────────────────────────────────────
FILE: experiments/exp_01_snr_validation.py
────────────────────────────────────────
Module docstring: "Experiment 1: Validates the theoretical SNR bound
empirically. Generates Figure 1 of the paper."

Script behavior when run directly:
  1. Sets random seed for reproducibility (seed=42)
  2. Defines configuration:
       d = 100  (gradient dimension)
       C = 1.0
       sigma_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
       k_values = [1, 2, 3, 5, 7, 10]  (with n=20)
       n_samples = 5000  (Monte Carlo samples per configuration)
  3. For each (sigma, k) combination:
     a. Create g_target = unit vector in R^d
     b. Create g_adv = C * g_target (already unit norm * C)
     c. Create DPMechanism(C=C, sigma=sigma)
     d. Sample n_samples outputs of M_DP(g_adv)
     e. Project each sample onto g_target direction
     f. Empirical SNR = mean(projections) / std(projections)
     g. Theoretical SNR = compute_attack_snr(k, n=20, 
                                              g_target_norm=C,
                                              sigma=sigma)
     Note: the projection step makes the comparison 
     sigma-and-n dependent correctly.
  4. Generate 2x2 matplotlib figure:
     - Plot 1: Empirical vs Theoretical SNR vs sigma (k=3 fixed)
     - Plot 2: Empirical vs Theoretical SNR vs k (sigma=0.1 fixed)
     - Plot 3: Relative error |empirical - theoretical| / theoretical
     - Plot 4: Distribution of M_DP(g_adv) projected on g_target
       for sigma=0.1, with vertical line at theoretical mean
  5. Save figure to paper/figures/fig_snr_validation.pdf
  6. Print summary table to stdout

────────────────────────────────────────
FILE: experiments/exp_02_mnist_attack.py
────────────────────────────────────────
Module docstring: "Experiment 2: Main DPAmplify attack on MNIST.
Demonstrates model accuracy degradation under the attack."

This is a simulation script (not using real Flower server/client
network). Instead it directly calls client fit() methods in a loop
to simulate federated rounds. This makes it faster to run and
easier to reproduce.

Script parameters (argparse):
  --n_clients: int = 20
  --n_byzantine: int = 3
  --clipping_C: float = 1.0
  --noise_sigma: float = 0.1
  --n_rounds: int = 100
  --T_est: int = 20
  --aggregator: str = "fedavg" (choices: fedavg, krum, trimmed_mean)
  --seed: int = 42
  --save_results: str = "results/exp_02_results.json"

Script behavior:
  1. Load MNIST, split into n_clients shards (IID split)
  2. Initialize n_clients - n_byzantine HonestClient instances
  3. Initialize n_byzantine DPAmplifyClient instances
     Target direction: unit vector e_1 (first coordinate)
  4. Initialize global MLP model
  5. For each round t in 1..n_rounds:
     a. Distribute current global model to all clients
     b. Each client calls fit() to get updated parameters
     c. Aggregate all updates with chosen aggregator
     d. Evaluate global model on MNIST test set
     e. Log: round, accuracy, loss, Byzantine phase (est/attack)
  6. Save results to JSON
  7. Print final accuracy with and without Byzantine attack
     (for comparison, also run a clean baseline without Byzantine)

────────────────────────────────────────
FILE: experiments/exp_03_evasion.py
────────────────────────────────────────
Module docstring: "Experiment 3: Tests whether DPAmplify gradients
evade Byzantine-robust aggregators (Krum, TrimmedMean)."

Script parameters (argparse):
  --aggregator: str (required, choices: krum, trimmed_mean)
  --n_clients: int = 20
  --n_byzantine: int = 3
  --n_rounds: int = 50
  --seed: int = 42

For Krum specifically, the script tests whether g_adv is selected
as the "trusted" gradient by checking if it appears in the Krum
output. This requires analyzing the distance structure of the
gradient set.

For TrimmedMean, the script checks whether the adversarial
coordinate values fall within the non-trimmed range.

Output: detection rate (fraction of rounds where Byzantine
gradient is flagged/excluded) and accuracy degradation.

────────────────────────────────────────
FILE: experiments/exp_04_adaptive_clipping.py
────────────────────────────────────────
Module docstring: "Experiment 4: Tests DPAmplify robustness under
adaptive clipping (Andrew et al. 2021 style), where C is updated
each round based on gradient norm quantiles."

Simulates adaptive clipping: C_t+1 = quantile_γ({‖g_i^t‖})
where γ = 0.5 (median adaptive clipping).

Shows that the passive estimator tracks C_t over time and
the attack remains effective even when C changes.

────────────────────────────────────────
FILE: countermeasures/randomized_clipping.py
────────────────────────────────────────
Module docstring: "Proposed countermeasure: randomized clipping
threshold. By randomly varying C each round from a distribution
U(C_min, C_max), the passive estimator cannot converge to a 
stable estimate of C, degrading attack effectiveness."

Function: randomized_clip(g: np.ndarray,
                           C_min: float,
                           C_max: float) -> Tuple[np.ndarray, float]:
  Samples C_t ~ U(C_min, C_max) and clips g to norm C_t.
  Returns (clipped_gradient, C_t_used)

Function: analyze_estimator_under_randomization(
    C_min: float, C_max: float,
    n_rounds: int = 100,
    n_trials: int = 50) -> dict:
  Runs the passive estimator under randomized clipping and
  measures estimation error. Returns dict with mean_error, std_error.

────────────────────────────────────────
FILE: countermeasures/gradient_auditor.py
────────────────────────────────────────
Module docstring: "Proposed countermeasure: statistical auditing
of gradient norms. Flags clients whose gradient norms cluster
suspiciously at exactly C (which is the signature of DPAmplify)."

Function: audit_gradient_norms(norms: List[float],
                                C_estimated: float,
                                tolerance: float = 0.01
                                ) -> List[bool]:
  Flags gradients where |‖g‖ - C| < tolerance.
  DPAmplify always submits gradients with norm exactly C.
  Returns list of bool (True = flagged as potentially Byzantine).

Function: norm_spike_detector(norm_history: List[float],
                               window: int = 10) -> bool:
  Detects unusual concentration of norms near a single value.
  Returns True if the distribution appears suspiciously peaked.

────────────────────────────────────────
FILE: tests/__init__.py
────────────────────────────────────────
Empty file.

────────────────────────────────────────
FILE: tests/test_dp_mechanism.py
────────────────────────────────────────
Complete pytest test file with:

test_clip_below_threshold:
  g = np.array([0.5, 0.0, 0.0])  (norm = 0.5 < C=1.0)
  result = DPMechanism(C=1.0, sigma=0.1).clip(g)
  assert np.allclose(result, g)  (no clipping applied)

test_clip_above_threshold:
  g = np.array([2.0, 0.0, 0.0])  (norm = 2.0 > C=1.0)
  result = DPMechanism(C=1.0, sigma=0.1).clip(g)
  assert abs(np.linalg.norm(result) - 1.0) < 1e-6

test_clip_exactly_at_threshold:
  g = np.array([1.0, 0.0, 0.0])  (norm = C = 1.0)
  result = DPMechanism(C=1.0, sigma=0.1).clip(g)
  assert np.allclose(result, g)

test_noise_zero_mean:
  np.random.seed(0)
  mech = DPMechanism(C=1.0, sigma=0.1)
  g = np.zeros(10)
  samples = mech.sample_outputs(g, n_samples=10000)
  assert np.abs(samples.mean()) < 0.01  (noise mean ≈ 0)

test_expected_output_equals_clip:
  mech = DPMechanism(C=1.0, sigma=0.1)
  g = np.array([2.0, 1.0, 0.5])
  assert np.allclose(mech.expected_output(g), mech.clip(g))

test_g_adv_norm_equals_C:
  from attack.gradient_optimizer import GradientOptimizer
  g_target = np.array([3.0, 4.0])  (norm = 5.0)
  opt = GradientOptimizer(g_target=g_target, C=1.0)
  g_adv = opt.compute_g_adv()
  assert abs(np.linalg.norm(g_adv) - 1.0) < 1e-6

test_g_adv_no_clipping:
  From above, clip(g_adv, C) should equal g_adv exactly.
  mech = DPMechanism(C=1.0, sigma=0.1)
  assert np.allclose(mech.clip(g_adv), g_adv)

test_estimator_ready_after_window:
  estimator = PassiveParameterEstimator(history_window=5)
  assert not estimator.is_ready()
  for norm in [0.9, 1.0, 1.1, 0.95, 1.05]:
      estimator.update(norm)
  assert estimator.is_ready()

test_invalid_C_raises:
  with pytest.raises(ValueError):
      DPMechanism(C=0.0, sigma=0.1)
  with pytest.raises(ValueError):
      DPMechanism(C=-1.0, sigma=0.1)

────────────────────────────────────────
FILE: notebooks/demo.ipynb
────────────────────────────────────────
Jupyter notebook with the following cells in order:

Cell 1 (Markdown): Title "DPAmplify — Interactive Demo"
  Brief description of what the notebook demonstrates.

Cell 2 (Markdown): "## 1. The DP Mechanism"

Cell 3 (Code): Import numpy, matplotlib, and DPMechanism.
  Create DPMechanism(C=1.0, sigma=0.1).
  Show clip() on a gradient with norm > C and norm < C.
  Print both results with their norms.

Cell 4 (Markdown): "## 2. The Adversarial Gradient"

Cell 5 (Code): Import GradientOptimizer.
  g_target = [1, 0, 0, ..., 0] (100-dim unit vector * 5)
  opt = GradientOptimizer(g_target, C=1.0)
  g_adv = opt.compute_g_adv()
  Print ‖g_adv‖, E[M_DP(g_adv)], and alignment with g_target.

Cell 6 (Markdown): "## 3. SNR Advantage"

Cell 7 (Code): Import snr_analysis functions.
  For n=20, sigma=0.1, plot SNR_attack vs k from 1 to 9.
  Add horizontal line at SNR=1 (minimum useful signal).
  Show the k threshold where SNR_attack > 1.

Cell 8 (Markdown): "## 4. Passive Parameter Estimation"

Cell 9 (Code): Simulate 25 rounds of norm observations.
  Show estimator converging to true C=1.0 and sigma=0.1.
  Plot estimated vs true values over rounds.

Cell 10 (Markdown): "## 5. Full Attack Simulation (Toy)"

Cell 11 (Code): Minimal simulation with n=5, k=1, 30 rounds.
  Use simple 2D gradient space for visualization.
  Plot the trajectory of the global model under attack.

────────────────────────────────────────
FILE: paper/dpamplify_arxiv.tex
────────────────────────────────────────
IEEE double-column LaTeX paper skeleton.

\documentclass[conference]{IEEEtran}

Packages: amsmath, amssymb, amsthm, algorithm2e, algpseudocode,
          graphicx, hyperref, cleveref, booktabs, xcolor

Theorem environments: theorem, lemma, corollary, definition,
                      proposition, remark

Paper structure with complete section headings and one
paragraph of [TODO: write content] per section:

\title{DPAmplify: Noise-Aware Byzantine Attacks Exploiting the
Analytical Structure of the Gaussian DP Mechanism in
Federated Learning}

\author{[Author] \\ [Institution] \\ [Email]}

\begin{abstract}
[TODO: 150-200 word abstract. Must include:
 - Problem: DP assumed to mitigate Byzantine attacks
 - Claim: this assumption is formally incorrect
 - Method: exploit E[M_DP(g_adv)] = g_adv
 - Result: formal SNR advantage k/√(n-k)
 - Impact: Krum/TrimmedMean cannot detect the attack]
\end{abstract}

Section I - Introduction: [TODO]
Section II - Background:
  \subsection{Federated Learning with Differential Privacy}
  \subsection{Byzantine Attacks in Federated Learning}
  
  Include the formal definition of M_DP as a displayed equation:
  \begin{equation}
    \mathcal{M}_{\text{DP}}(\mathbf{g}) = 
    \text{clip}(\mathbf{g}, C) + \boldsymbol{\xi}, \quad
    \boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
  \end{equation}
  
  And the clipping operation:
  \begin{equation}
    \text{clip}(\mathbf{g}, C) = \mathbf{g} \cdot 
    \min\!\left(1, \frac{C}{\|\mathbf{g}\|_2}\right)
  \end{equation}

Section III - Threat Model: [TODO]

Section IV - The DPAmplify Attack:
  \subsection{Phase 1: Passive Parameter Estimation}
  \subsection{Phase 2: Adversarial Gradient Construction}
  
  Include the main theorem as a \begin{theorem} block:
  \begin{theorem}[SNR Advantage of DPAmplify]
  Let ... [TODO: fill in theorem statement from theorem_snr.tex]
  \end{theorem}
  \begin{proof}
  [TODO: proof sketch]
  \end{proof}
  
  Include the attack algorithm as an \begin{algorithm} block.

Section V - Evaluation: [TODO]
  \subsection{Experimental Setup}
  \subsection{SNR Validation}
  \subsection{Attack on MNIST}
  \subsection{Evasion of Byzantine-Robust Aggregators}
  \subsection{Robustness under Adaptive Clipping}

Section VI - Countermeasures: [TODO]
  \subsection{Randomized Clipping Threshold}
  \subsection{Gradient Norm Auditing}

Section VII - Related Work:
  Must cite and differentiate from:
  - Robust-HDP (Malekmohammadi et al., ICML 2024)
  - LIE (Baruch et al., NeurIPS 2019)
  - MinMax/MinSum (Shejwalkar & Houmansadr, USENIX 2021)
  - FLTrust (Cao et al., NDSS 2022)
  [TODO: write differentiation paragraphs]

Section VIII - Conclusion: [TODO]

\bibliography{references}
\bibliographystyle{IEEEtran}

Also create paper/references.bib with BibTeX entries for
the four papers listed above plus:
- Abadi et al. 2016 (DP-SGD)
- McMahan et al. 2017 (FedAvg)
- Blanchard et al. 2017 (Krum)
- Yin et al. 2018 (TrimmedMean)
Use realistic BibTeX entries (correct authors, years, venues)
— only cite papers verified to exist.

══════════════════════════════════════════════════════════
FINAL INSTRUCTIONS
══════════════════════════════════════════════════════════

After generating all files:
1. Print the complete directory tree
2. Confirm that every Python file has been verified to
   be syntactically correct
3. List any assumptions made where the specification
   was ambiguous
4. Flag any place where a Flower API call may need
   version-specific adjustment (flwr 1.8.x API)
