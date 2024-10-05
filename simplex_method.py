import numpy as np

simplex_input = (
    [-6, -10, -4],          # C - objective func. coefficients
    [
        [1, 1, 1],
        [1, 1, 0],
        [1, 2, 0],
    ],               # A - constraint coefficients
    [1000, 500, 700],         # b - rhs of constraints
    1e-4,            # e - precision
    "min"            # max or min
)

np.set_printoptions(suppress=True)

def simplex(input_data: tuple = simplex_input) -> list[int] | str:

    # parsing input data
    c_objective: np.ndarray = np.array(input_data[0], dtype=float)
    constraints: np.ndarray = np.array(input_data[1], dtype=float)
    rhs: np.ndarray = np.array(input_data[2], dtype=float)
    precision: float = input_data[3]
    max: bool = True if input_data[4] == "max" else False

    n_vars: int = len(c_objective)
    n_constraints: int = len(constraints)

    # input validation
    if any(len(input) == 0 for input in input_data[:-2]) or input_data[4] not in ("max", "min"):
        return "Error: Input malformation!"
    if any(len(constraint) != n_vars for constraint in constraints):
        return "Error: Number of variables in constraints and objective function is different!"
    if len(rhs) != n_constraints:
        return "Error: Number of rhs values and the number of constraints are different!"
    if all(x < 0 for x in rhs):
        return "Unbounded solution!"

    # printing the optimization problem
    str_problem: str = ("max" if max else "min") + f" z = {c_objective[0]} * x1"
    for i, coefficient in enumerate(c_objective[1:]):
        if coefficient == 0:
            term = ""
        elif coefficient < 0:
            term = f" - {abs(coefficient)} * x{i + 2}"
        else:
            term = f" + {coefficient} * x{i + 2}"
        str_problem += term
    str_problem = str_problem[:-3] + "\nsubject to the constraints:\n"
    for i, coefficients in enumerate(constraints):
        for j, coefficient in enumerate(coefficients):
            str_problem += f"{coefficient} * x{j + 1} + "
        str_problem = str_problem[:-3] + f" <= {rhs[i]}\n"
    print(str_problem)
    
    # initialize table with zeros
    table: np.ndarray = np.zeros((n_constraints + 1, n_vars + n_constraints + 1))
    # fill in z-row
    table[0, :n_vars] = -c_objective if max else c_objective
    # fill in basic variable rows
    table[1:, :-(1 + n_constraints)] = constraints
    # fill in rhs
    table[1:, -1] = rhs

    
    # simplex loop
    while np.any(table[0, :-1] < -precision):
        print(table)
        # index of the minimal element in z-row
        pivot_col: int = np.argmin(table[0, :-1])
        
        # contains ratios (rhs/pivot_column_element)
        ratios: np.ndarray = table[1:, -1] / table[1:, pivot_col]
        
        # problem is unbounded if all elements are < 0 or are infinite
        if np.all((ratios <= precision) | np.isinf(ratios)):
            return "Unbounded problem!"
        
        # ignore negative elements (for the next line)
        ratios[ratios <= 0] = np.inf
        
        # ratios are built starting from the second row, so add one
        pivot_row: int = np.argmin(ratios) + 1

        pivot_elem: float = table[pivot_row, pivot_col]
        
        # normalize pivot row
        table[pivot_row] /= pivot_elem
        
        for i in range(n_constraints + 1):
            if i != pivot_row:
                table[i] -= table[i, pivot_col] * table[pivot_row]
    return table
    
    
    
print(simplex())