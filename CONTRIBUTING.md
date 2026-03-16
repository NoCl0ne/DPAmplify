# Contributing to DPAmplify

DPAmplify is an active research project. Contributions are welcome
under the guidelines below.

## Status

This repository accompanies a paper currently under review.
Full public release of certain modules is scheduled 90 days
after paper publication.

## Bug Reports

Please open a GitHub Issue with:
- A minimal reproducible example
- Your Python and dependency versions (`pip freeze`)
- The complete traceback

## Pull Requests

Pull requests are welcome **after paper publication**.
Until then, please open an Issue to discuss proposed changes
before investing time in an implementation.

When submitting a PR:
1. Fork the repository and create a feature branch.
2. Ensure `pytest tests/ -v` passes before opening the PR.
3. Add or update tests for any new behavior.
4. Keep commits focused — one logical change per commit.

## Ethical Policy

This code is intended exclusively for **defensive security research**:
- Studying vulnerabilities to build better defenses
- Academic reproduction of published results
- Development of countermeasures

**Do not** use this code to attack real federated learning systems
or any system without explicit written authorization from its owner.
The authors accept no liability for misuse.

## Code Style

- Python 3.11+, formatted with `black` (default settings)
- Type hints on all public function signatures
- Docstrings in Google style

## Contact

Open a GitHub Issue for questions related to the code or paper.
