import numpy as np

simplex_input = (
    [6, 9],          # C - objective func. coefficients
    [[2, 3],[1, 1]], # A - constraint coefficients
    [12, 5],         # b - rhs of constraints
    1e-4,            # e - precision
    True             # max or min - true or false
)


def simplex(input_data: tuple = simplex_input) -> list[int] | str:

    # parsing input data
    c_objective: np.ndarray[int] = np.array(input_data[0], dtype=float)
    constraints: np.ndarray[int] = np.array(input_data[1], dtype=float)
    rhs: np.ndarray[int] = np.array(input_data[2], dtype=float)
    precision: float = input_data[3]
    max: bool = input_data[4]

    n_vars: int = len(c_objective)
    n_constraints: int = len(constraints)

    # input validation
    if (any(len(input) == 0 for input in input_data[:-2])):
        return "Error: Input malformation!"
    if (any(len(constraint) != n_vars for constraint in constraints)):
        return "Error: Number of variables in constraints and objective function is different!"
    if (len(rhs) != n_constraints):
        return "Error: Number of rhs values and the number of constraints are different!"
    if (any(x < 0 for x in rhs)):
        return "Error: A rhs value is negative!"

    # printing the optimization problem
    str_problem: str = ("max" if max else "min") + " z = "
    for i, coefficient in enumerate(c_objective):
        str_problem += f"{coefficient} * x{i + 1} + "
    str_problem = str_problem[:-3] + "\nsubject to the constraints:\n"
    for i, coefficients in enumerate(constraints):
        for j, coefficient in enumerate(coefficients):
            str_problem += f"{coefficient} * x{j + 1} + "
        str_problem = str_problem[:-3] + f" <= {rhs[i]}\n"
    print(str_problem)
    
    # initialize table with zeros
    table: np.ndarray = np.zeros((n_constraints + 1, n_vars + n_constraints + 1))
    # fill in z-row
    table[0, :n_vars] = -c_objective
    # fill in basic variable rows
    table[1:, :-(1 + n_vars)] = constraints
    # fill in rhs
    table[1:, -1] = rhs



    while (all(table[0, :-1] > 0)): # simplex loop
        pass

    print(table)

simplex()