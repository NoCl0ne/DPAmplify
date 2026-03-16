# DPAmplify

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status: Research in Progress](https://img.shields.io/badge/status-research%20in%20progress-orange.svg)]()
[![arXiv: forthcoming](https://img.shields.io/badge/arXiv-forthcoming-red.svg)]()

## DPAmplify: Noise-Aware Byzantine Attacks Exploiting the Analytical 
## Structure of the Gaussian DP Mechanism in Federated Learning

---

### Overview

DPAmplify is a research project that identifies and demonstrates a 
previously uncharacterized attack vector in differentially-private 
federated learning (DP-FL).

The common assumption in the federated learning security literature 
is that differential privacy (DP) noise reduces the effectiveness of 
Byzantine attacks by masking malicious gradient updates.  
**DPAmplify inverts this assumption.**

We show that a Byzantine participant who knows — or can passively 
estimate — the clipping threshold `C` and noise standard deviation `σ` 
of the Gaussian DP mechanism can construct adversarial gradients whose 
**expected post-noise value converges coherently toward an adversarial 
target direction**, achieving a formal signal-to-noise ratio (SNR) 
advantage over honest participants, even under Byzantine-robust 
aggregation rules such as Krum and TrimmedMean.

This work challenges the widely held belief that combining DP with 
Byzantine-robust aggregation provides defense-in-depth in federated 
learning.

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

A Byzantine participant sets `g_adv = C · (g_target / ‖g_target‖₂)`.  
Because `‖g_adv‖₂ = C` exactly, the clipping operation is a no-op:
```
E[M_DP(g_adv)] = g_adv   →   aligned with g_target
```

With `k` Byzantine clients repeating this over `T` rounds, the 
adversarial signal accumulates coherently while honest gradients 
undergo random-walk cancellation, yielding a formal SNR advantage:
```
SNR_attack = k · ‖g_target‖ / (σ · √(n − k))
```

---

### Differentiation from Prior Work

| Work | What it does | Why DPAmplify is different |
|------|-------------|---------------------------|
| **Robust-HDP** (Malekmohammadi et al., ICML 2024) | Noise-aware *aggregation* — server estimates DP noise to improve utility | Defensive, server-side. DPAmplify is an *offensive* client-side attack exploiting the same noise structure |
| **LIE** (Baruch et al., NeurIPS 2019) | Crafts malicious gradients using statistics of honest updates to evade detection | Does not consider the DP mechanism. DPAmplify specifically exploits the analytical structure of the Gaussian mechanism |
| **MinMax / MinSum** (Shejwalkar & Houmansadr, USENIX 2021) | Optimizes perturbation to evade aggregator metrics | No DP-awareness. DPAmplify uses `E[M_DP(·)]` as the explicit optimization objective |
| **FLTrust** (Cao et al., NDSS 2022) | Uses a server-side root dataset for gradient scoring | Vulnerable when Byzantine clients can passively estimate the root gradient direction |

**Key novelty confirmed by literature search (March 2026):**  
No existing paper uses `E[M_DP(g_adv)]` as the optimization objective 
for a Byzantine attack, nor formally proves the SNR advantage 
`k/√(n−k)` in DP-FL settings.

---

### Threat Model

- **Setting:** Cross-silo federated learning without Secure Aggregation (SecAgg)
- **Attacker:** `k` Byzantine clients (`k < n/2`) under honest-but-curious server
- **Attacker knowledge:** Black-box access to the FL system; can observe 
  aggregated gradient norms across rounds
- **Attacker capability:** Controls gradient updates submitted by `k` clients; 
  cannot modify server-side aggregation or other clients' updates
- **Attack phases:**
  - **Phase 1 — Passive estimation:** Byzantine clients behave honestly 
    while observing aggregated gradient norms to estimate `C` and `σ` 
    without any detectable deviation from normal behavior
  - **Phase 2 — Attack:** Byzantine clients submit `g_adv` optimized 
    to accumulate toward `g_target` in expectation

**Out of scope:** Settings with SecAgg, trusted execution environments, 
or cryptographic gradient verification. These are documented as explicit 
limitations in the paper.

---

### Repository Structure
```
dpamplify/
│
├── theory/                         # Formal mathematical foundations
│   ├── dp_mechanism.py             # Gaussian DP mechanism implementation
│   ├── snr_analysis.py             # Theoretical SNR computation
│   └── proofs/
│       └── theorem_snr.tex         # Formal LaTeX proof of SNR theorem
│
├── attack/                         # Attack implementation
│   ├── parameter_estimator.py      # Passive estimation of C and σ
│   ├── gradient_optimizer.py       # Computation of optimal g_adv
│   └── byzantine_client.py         # DPAmplify Byzantine client (Flower)
│
├── fl_system/                      # Federated learning infrastructure
│   ├── server.py                   # FL server with DP and logging
│   ├── honest_client.py            # Honest client with Opacus DP-SGD
│   └── aggregators/
│       ├── fedavg.py               # FedAvg baseline
│       ├── krum.py                 # Krum (Blanchard et al. 2017)
│       └── trimmed_mean.py         # TrimmedMean (Yin et al. 2018)
│
├── experiments/                    # Reproducible experiments
│   ├── exp_01_snr_validation.py    # Empirical SNR vs theoretical bound
│   ├── exp_02_mnist_attack.py      # Main attack on MNIST
│   ├── exp_03_evasion.py           # Evasion of Krum and TrimmedMean
│   └── exp_04_adaptive_clipping.py # Robustness under adaptive clipping
│
├── countermeasures/                # Proposed defenses
│   ├── randomized_clipping.py      # Randomized clipping threshold
│   └── gradient_auditor.py         # Statistical auditing of gradients
│
├── tests/                          # Unit and integration tests
│   └── test_dp_mechanism.py
│
├── notebooks/
│   └── demo.ipynb                  # Interactive demonstration
│
├── paper/
│   ├── dpamplify_arxiv.tex         # arXiv preprint (IEEE format)
│   ├── references.bib              # BibTeX bibliography
│   └── figures/                    # Generated figure scripts
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
# 1. Clone the repository
git clone https://github.com/[YOUR_USERNAME]/dpamplify.git
cd dpamplify

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .

# 5. Verify installation
python -c "from theory.dp_mechanism import DPMechanism; print('OK')"
pytest tests/ -v
```

---

### Quick Start

**Verify the theoretical SNR bounds empirically:**
```bash
python experiments/exp_01_snr_validation.py
```

Generates `paper/figures/fig_snr_validation.pdf` comparing empirical 
SNR against the theoretical bound across values of `σ` and `k`.

**Run the main attack on MNIST:**
```bash
python experiments/exp_02_mnist_attack.py \
    --n_clients 20 \
    --n_byzantine 3 \
    --clipping_C 1.0 \
    --noise_sigma 0.1 \
    --n_rounds 100 \
    --aggregator fedavg
```

**Test evasion of robust aggregators:**
```bash
python experiments/exp_03_evasion.py --aggregator krum
python experiments/exp_03_evasion.py --aggregator trimmed_mean
```

**Interactive demo:**
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
| Noise multiplier σ | 0.1 | Varied in ablation study |
| Estimation rounds T_est | 20 | Passive phase duration |
| Attack rounds T_attack | 100 | Active phase duration |
| Model | MLP 784→128→10 | MNIST classification |
| Local epochs | 2 | Per round |
| Batch size | 64 | — |
| Aggregator (baseline) | FedAvg | Also tested: Krum, TrimmedMean |

---

### Scope and Limitations

This research applies to federated learning deployments that:

- Use the Gaussian DP mechanism (DP-SGD with L2 clipping + Gaussian noise)
- Do **not** use Secure Aggregation (SecAgg)
- Do **not** use cryptographic gradient verification
- Have a static or slowly adaptive clipping threshold

The attack does **not** apply to settings protected by SecAgg, trusted 
hardware enclaves, or cryptographic gradient attestation protocols.  
These are documented explicitly as out-of-scope in Section VI of the paper.

---

### Ethical Statement

This research is conducted to expose a structural vulnerability in 
differentially-private federated learning and to motivate the 
development of stronger defenses.

- All experiments are conducted exclusively in isolated simulation 
  environments
- No real-world federated learning systems were targeted
- The paper includes a **Countermeasures** section of equal prominence 
  to the attack description
- Responsible disclosure to maintainers of Flower and Opacus will occur 
  prior to public release of the full attack implementation
- The file `attack/byzantine_client.py` will be released 90 days after 
  paper publication to allow framework maintainers to implement mitigations



