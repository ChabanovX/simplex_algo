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
