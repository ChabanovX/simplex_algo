# Simplex Optimization Method in Python

This repository contains an implementation of the **Simplex Method** for solving linear programming problems, written in **Python**. The Simplex algorithm is a widely used optimization technique to find the optimal solution for linear constraints and objective functions.

## Features

- High-performance implementation using Python
- Solves linear programming problems with ease
- Easily extendable to handle custom constraints and objective functions
- Designed for educational purposes as well as practical use in optimization tasks

## Requirements

Ensure you have Python installed. You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repo
```bash
git clone https://github.com/yourusername/simplex-python.git
cd simplex-python
```
2. Run the Simplex algorithm with your own parameters:
```python
from simplex import simplex
c = [-3.0, -2.0]   # Cost function coefficients
A = [[1.0, 1.0], [2.0, 1.0], [0.0, 1.0]]  # Constraint matrix
b = [4.0, 5.0, 1.0]  # RHS of constraints
solution, optimal_value = simplex(c, A, b)
print("Optimal solution:", solution)
print("Optimal value:", optimal_value)
```

## Contributing
Feel free to submit issues or pull requests if you’d like to contribute or improve this implementation.

## License
MIT License

