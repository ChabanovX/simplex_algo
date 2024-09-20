# Simplex Optimization Method in Julia

This repository contains an implementation of the **Simplex Method** for solving linear programming problems, written in **Julia**. The Simplex algorithm is a widely used optimization technique to find the optimal solution for linear constraints and objective functions.

## Features

- High-performance implementation using Julia
- Solves linear programming problems with ease
- Easily extendable to handle custom constraints and objective functions
- Designed for educational purposes as well as practical use in optimization tasks

## Usage

### Requirements

Ensure you have Julia installed. You can download it from [Julia's official website](https://julialang.org/downloads/).

### Running the Code

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/simplex-julia.git
   cd simplex-julia

2.	Run the Simplex algorithm with your own parameters:
     ```Julia
      include("simplex.jl")
      c = [-3.0, -2.0]   # Cost function coefficients
      A = [1.0 1.0; 2.0 1.0; 0.0 1.0]  # Constraint matrix
      b = [4.0, 5.0, 1.0]  # RHS of constraints
      solution, optimal_value = simplex(c, A, b)
      println("Optimal solution: ", solution)
      println("Optimal value: ", optimal_value)
     ```

### Licence
MIT Licence
  

  
