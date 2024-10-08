from simplex_method import simplex, print_simplex_result

lpp = {
    "max": True,          # max or min - True or False
    "C": [10, 6.5],       # C - objective function coefficients
    "A": [                # A - constraint coefficients matrix
        [20.3, 10],
        [40, 10],
    ],                    
    "b": [1200, 1600],    # b - rhs of constraints
    "e": 1e-4             # e - precision
}

res = simplex(lpp)
print_simplex_result(res)
