import numpy as np
import sys

simplex_input = (
    [6, 9],          # C - objective func. coefficients
    [
        [2, 3],
        [1, 1],
    ],               # A - constraint coefficients
    [12, 5],         # b - rhs of constraints
    1e-4,            # e - precision
    "max"            # max or min
)

np.set_printoptions(suppress=True)
np.seterr(divide='ignore')

def simplex(input_data: tuple = simplex_input) -> list | str:

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
    str_problem: str = f"problem: {"max" if max else "min"} z = {c_objective[0]} * x1"
    for i, coefficient in enumerate(c_objective[1:]):
        if coefficient == 0:
            term = ""
        elif coefficient < 0:
            term = f" - {abs(coefficient)} * x{i + 2}"
        else:
            term = f" + {coefficient} * x{i + 2}"
        str_problem += term
    str_problem += "\nsubject to the constraints:\n"
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
        # index of the minimal element in z-row
        pivot_col: int = np.argmin(table[0, :-1])
        
        # contains ratios (rhs/pivot_column_element)
        ratios: np.ndarray = table[1:, -1] / table[1:, pivot_col]
        
        # problem is unbounded if all elements are < 0 or are infinite
        if np.all((ratios <= precision) | np.isinf(ratios)):
            return "Unbounded problem!"
        
        # ignore negative elements (for the next line)
        ratios[abs(ratios) < precision] = np.inf
        
        # ratios are built starting from the second row, so add one
        pivot_row: int = np.argmin(ratios) + 1

        pivot_elem: float = table[pivot_row, pivot_col]
        
        # normalize pivot row
        table[pivot_row] /= pivot_elem
        
        # do this idk
        for i in range(n_constraints + 1):
            if i != pivot_row:
                table[i] -= table[i, pivot_col] * table[pivot_row]
    
    solution = []
    
    # collect solution
    for i in range(n_vars):
        # find a basic variable
        if (1 - precision < np.sum(table[1:, i])  <  1 + precision) and (np.max(table[1:, i]) == 1):
            basic_row: int = np.where(table[:, i] == 1)[0][0]
            basic_value: float = table[basic_row, -1]
            # place basic variable's index and its value
            solution.append((i + 1, float(basic_value)))
    # if finidng min, multiply by -1
    if not max:
        table[0] *= -1
    # place the z value in the end
    solution.append(float(table[0, -1]))
    return solution
    
    
res = simplex()

if isinstance(res, str):
    print(res, file=sys.stderr)
else:
    output_str: str = f"solution: z = {res.pop()}, where\n"
    for i, value in res:
        output_str += f"x{i} = {value}, "
    print(output_str)
