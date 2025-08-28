# ✅ TODO — Vanilla Pricer (Personal Quant Dev Project)

This document tracks development tasks to improve and professionalize the project.  
Each task helps strengthen Python, OOP, testing, CI/CD, and quantitative finance skills.

---

## 📦 CORE FEATURES

- [ ] Add **put option pricing** to all models (Black-Scholes, Binomial Tree, Monte Carlo)
- [ ] Implement **Greeks** in Black-Scholes (Delta, Gamma, Vega, Theta, Rho)
- [ ] Add **early exercise** for American options in the Binomial Tree model
- [ ] Improve Monte Carlo with:
  - [ ] Antithetic variates
  - [ ] Control variates
  - [ ] Convergence diagnostics

---

## 🧪 TESTING & COVERAGE

- [ ] Add tests for edge cases (T=0, sigma=0, deep ITM/OTM, etc.)
- [ ] Use `pytest-cov` to track code coverage
- [ ] Ensure 90%+ test coverage

---

## ⚙️ TOOLING & CI/CD

- [ ] Add a `pyproject.toml` or `setup.py` for local installation (`pip install .`)
- [ ] Add GitHub Actions to:
  - [ ] Run tests on push/PR
  - [ ] Enforce code formatting (e.g. `black`, `ruff`)
- [ ] Add badges (build passing, coverage) to `README.md`

---

## 📊 VISUALIZATIONS & ANALYSIS

- [ ] Create a Jupyter Notebook to:
  - [ ] Show pricing vs parameters (S, T, sigma, r)
  - [ ] Plot Binomial Tree for small `n`
  - [ ] Plot Monte Carlo simulation paths
- [ ] Use `matplotlib`, `seaborn`, or `plotly` for visualization

---

## 📚 DOCUMENTATION

- [ ] Improve `README.md` with:
  - [ ] Example usage
  - [ ] Installation guide
  - [ ] Mathematical background
- [ ] (Optional) Use **Sphinx** or **MkDocs** for auto-generated doc site
- [ ] Add docstrings to all classes and methods

---

## 🧠 EXTENSIONS (Advanced)

- [ ] Add support for dividends in Black-Scholes
- [ ] Implement volatility smile visualization
- [ ] Add calibration tools (from market data)
- [ ] Add pricing for exotic options (e.g. digital, barrier, Asian)

---

## ✨ NICE TO HAVE

- [ ] Create CLI (`pricer-cli`) to price options from terminal
- [ ] Package the project for PyPI (learning purpose)
- [ ] Dockerize the project for reproducibility
