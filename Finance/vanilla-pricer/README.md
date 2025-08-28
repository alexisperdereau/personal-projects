# Vanilla Option Pricer

## Overview
The **Vanilla Option Pricer** is a personal project aimed at implementing different option pricing models. The library includes several popular models for pricing European call options, such as the **Black-Scholes model**, **Binomial Tree method**, and **Monte Carlo simulation**. This project focuses on implementing these models in Python while documenting the code with clear and comprehensive comments for better understanding.

## Features
- **Black-Scholes Model**: A closed-form solution for European call options.
- **Binomial Tree Model**: A numerical method using a discrete time step to model option prices.
- **Monte Carlo Simulation**: A probabilistic approach to simulate the paths of the underlying asset and calculate the option price.
- **Clear Documentation**: Code is commented thoroughly, with mathematical formulas included, making it easy to understand for both beginners and professionals.

## Purpose
This project is intended to be used as an educational tool. The main goal is to provide a clear and structured implementation of various option pricing methods, with detailed explanations of the mathematical models behind each one. It is suitable for anyone interested in learning how to implement financial models in Python, particularly in the context of options pricing.

## Getting Started

### Prerequisites
You need to have Python installed on your machine. You can install it from [python.org](https://www.python.org/downloads/).

Additionally, this project uses the following Python libraries:
- `numpy`
- `scipy`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/alexisperdereau/vanilla-pricer.git


### Structure

vanilla-pricer/
│
├── vanilla_pricer/                # Main library code
│   ├── __init__.py                # Initialize the package
│   ├── option.py                  # Option class to define option parameters
│   ├── black_scholes.py           # Black-Scholes pricing model
│   ├── binomial_tree.py           # Binomial Tree pricing model
│   ├── monte_carlo.py             # Monte Carlo simulation model
│
├── requirements.txt               # Required dependencies
├── setup.py                       # Setup script for package installation
├── README.md                      # This file
└── tests/                         # Unit tests for validation
    ├── test_black_scholes.py      # Unit test for Black-Scholes
    ├── test_binomial_tree.py      # Unit test for Binomial Tree model
    └── test_monte_carlo.py        # Unit test for Monte Carlo simulation
