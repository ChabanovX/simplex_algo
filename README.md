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
# main.py
from simplex_method import simplex, print_simplex_result

lpp = {
    "C": [6, 9],          # C - objective function coefficients list
    "A": [                # A - constraint coefficients matrix
        [2, 3],
        [1, 1],
    ],                    
    "b": [12, 5],         # b - rhs of constraints list
    "e": 1e-4,            # e - precision float
    "max": True           # max or min - True or False
}

res = simplex(lpp)
print_simplex_result(res)
```

## Contributing
Feel free to submit issues or pull requests if you’d like to contribute or improve this implementation.

## License
MIT License

