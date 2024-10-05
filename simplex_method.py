import numpy as np
import sys

np.set_printoptions(suppress=True)
np.seterr(divide='ignore')

def simplex(lpp: dict) -> list | str:
    # parsing input data
    c_objective: np.ndarray = np.array(lpp["C"], dtype=float)
    constraints: np.ndarray = np.array(lpp["A"], dtype=float)
    rhs: np.ndarray = np.array(lpp["b"], dtype=float)
    precision: float = lpp["e"]
    max: bool = lpp["max"]

    check_lpp(lpp)

    print_lpp(lpp)
    
    n_constraints: int = len(constraints)
    n_vars: int = len(c_objective)
    # initialize table with zeros
    table: np.ndarray = np.zeros((n_constraints + 1, n_vars + n_constraints + 1))
    # fill in z-row
    table[0, :n_vars] = -c_objective if max else c_objective
    # fill in basic variable rows
    table[1:, :-(1 + n_constraints)] = constraints
    # fill in rhs
    table[1:, -1] = rhs
    
    # simplex loop while there are negative entries in the z-row
    while np.any(table[0, :-1] < -precision):
        # index of the minimal element in z-row
        pivot_col: int = np.argmin(table[0, :-1])
        
        # find ratios (rhs/pivot_column_element)
        ratios: np.ndarray = table[1:, -1] / table[1:, pivot_col]
        
        # problem is unbounded if all elements are < 0 or infinite
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

def check_lpp(lpp: dict) -> None:
    c_objective, constraints, rhs, precision, __ = lpp.values()
    n_vars: int = len(c_objective)
    n_constraints: int = len(constraints)

    assert all(len(constraint) == n_vars for constraint in constraints),\
        "Error:\nMalformed input: Number of variables in constraints and objective function is different!"
    assert len(rhs) == n_constraints,\
        "Error:\n Malformed input: Number of rhs values and the number of constraints are different!"
    assert all(x > precision for x in rhs),\
        "Unbounded solution!"

def print_lpp(lpp: dict) -> None:
    c_objective, constraints, rhs, _, max = lpp.values()
    print(c_objective, constraints, rhs, _, max)
    str_problem: str = f"problem:\n{"max" if max else "min"} z = {c_objective[0]:g} * x1"
    for i, coefficient in enumerate(c_objective[1:]):
        if coefficient == 0:
            term = ""
        elif coefficient < 0:
            term = f" - {abs(coefficient):g} * x{i + 2}"
        else:
            term = f" + {coefficient:g} * x{i + 2}"
        str_problem += term
    str_problem += "\nsubject to the constraints:\n"
    for i, coefficients in enumerate(constraints):
        for j, coefficient in enumerate(coefficients):
            str_problem += f"{coefficient:g} * x{j + 1} + "
        str_problem = str_problem[:-3] + f" <= {rhs[i]:g}\n"
    print(str_problem[:-1])


def print_simplex_result(res: str | list) -> None:
    if isinstance(res, str):
        print(res, file=sys.stderr)
    else:
        output_str: str = f"solution:\nz = {res.pop():g}"
        if len(res) == 0:
            print(output_str)
            return
        output_str += ',\n'
        for i, value in res:
            output_str += f"x{i} = {value:g},\n"
        output_str = output_str[:-2]
        print(output_str)