from simplex_method import simplex, print_simplex_result

lpp = (
    [6, 9],          # C - objective function coefficients
    [
        [2, 3],
        [1, 1],
    ],               # A - constraint coefficients
    [12, 5],         # b - rhs of constraints
    1e-4,            # e - precision
    "max"            # max or min
)

res = simplex(lpp)
print_simplex_result(res)
