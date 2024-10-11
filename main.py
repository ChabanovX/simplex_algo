from simplex_method import simplex, print_simplex_result

lpp = {
    "max": True,          # max or min - True or False
    "C": [-2, 2, -6],       # C - objective function coefficients
    "A": [                # A - constraint coefficients matrix
        [2, 1, -2],
        [1, 2, 4],
        [1, -1, 2]
    ],                    
    "b": [24, 23, 10],    # b - rhs of constraints
    "e": 1e-4             # e - precision
}

res = simplex(lpp)
print_simplex_result(res)
