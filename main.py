from simplex_method import simplex, print_simplex_result

lpp = {
    "max": True,          # max or min - True or False
    "C": [5, 4],       # C - objective function coefficients
    "A": [                # A - constraint coefficients matrix
        [6, 4],
        [1, 2],
        [-1, 1],
    ],                    
    "b": [24, 6, 1],    # b - rhs of constraints
    "e": 1e-4             # e - precision
}

res = simplex(lpp)
print_simplex_result(res)
