from simplex_method import simplex, print_simplex_result

lpp = {
    "max": True,          # max or min - True or False
    "C": [3, 9],       # C - objective function coefficients
    "A": [                # A - constraint coefficients matrix
        [1, 4],
        [1, 2],
    ],                    
    "b": [8, 4],    # b - rhs of constraints
    "e": 1e-4             # e - precision
}

res = simplex(lpp)
print_simplex_result(res)
