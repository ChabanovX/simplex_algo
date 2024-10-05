from simplex_method import simplex, print_simplex_result

lpp = {
    "C": [10, 4],          # C - objective function coefficients list
    "A": [                # A - constraint coefficients matrix
        [20, 10],
        [40, 10],
    ],                    
    "b": [1200, 1600],         # b - rhs of constraints list
    "e": 1e-4,            # e - precision float
    "max": True           # max or min - True or False
}

res = simplex(lpp)
print_simplex_result(res)
