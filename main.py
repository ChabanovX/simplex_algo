from simplex_method import simplex, print_simplex_result

lpp = {
    "max": True,          # max or min - True or False
    "C": [2, 1],       # C - objective function coefficients
    "A": [                # A - constraint coefficients matrix
        [1, -1],
        [2, 0],
    ],                    
    "b": [10, 40],    # b - rhs of constraints
    "e": 1e-4             # e - precision
}

res = simplex(lpp)
print_simplex_result(res)
