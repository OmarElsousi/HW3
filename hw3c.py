import numpy as np
import matrixOperations as mo
import numericalMethods as nm
import DoolittleMethod as dm  # Importing BackSolve from here


def is_symmetric(A):
    """
    Check if a given matrix A is symmetric.
    A matrix is symmetric if it is equal to its transpose (A == A^T).

    Parameters:
        A (numpy.ndarray): The input matrix.

    Returns:
        bool: True if A is symmetric, False otherwise.
    """
    return np.allclose(A, np.transpose(A))


def is_positive_definite(A):
    """
    Check if a given matrix A is positive definite.
    A matrix is positive definite if all its eigenvalues are positive.

    Parameters:
        A (numpy.ndarray): The input matrix.

    Returns:
        bool: True if A is positive definite, False otherwise.
    """
    return np.all(np.linalg.eigvals(A) > 0)


def cholesky_factorization(A):
    """
    Perform Cholesky decomposition of a given matrix A.
    This decomposition expresses A as the product of a lower triangular matrix L
    and its transpose (A = L * L^T).

    Parameters:
        A (numpy.ndarray): The input symmetric, positive-definite matrix.

    Returns:
        numpy.ndarray: The lower triangular matrix L such that A = L * L^T.
    """
    n = len(A)
    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - sum_k)
            else:
                L[i][j] = (A[i][j] - sum_k) / L[j][j]

    return L


def cholesky_solve(A, b):
    """
    Solve the system Ax = b using Cholesky decomposition.
    First, it decomposes A into L * L^T, then solves for x.

    Parameters:
        A (numpy.ndarray): The input symmetric, positive-definite matrix.
        b (numpy.ndarray): The right-hand side vector.

    Returns:
        numpy.ndarray: The solution vector x.
    """
    L = cholesky_factorization(A)
    y = dm.BackSolve(L, b, UT=False)  # Solve Ly = b (Lower Triangular)
    x = dm.BackSolve(np.transpose(L), y, UT=True)  # Solve L^T x = y (Upper Triangular)
    return x


def solve_system(Aaug):
    """
    Solve the system of linear equations given in augmented matrix form.
    It determines the appropriate method (Cholesky or Doolittle) based on matrix properties.

    Parameters:
        Aaug (numpy.ndarray): The augmented matrix [A|b] where A is the coefficient matrix and b is the RHS vector.

    Returns:
        numpy.ndarray: The solution vector x.
    """
    A, b = mo.separateAugmented(Aaug)  # Separate augmented matrix into A and b

    if is_symmetric(A) and is_positive_definite(A):
        print("\nUsing Cholesky method:")
        x = cholesky_solve(A, b)
    else:
        print("\nUsing Doolittle method:")
        x = dm.Doolittle(Aaug)

    return x


def main():
    """
    Main function to define and solve two example systems of linear equations.
    It applies the appropriate method (Cholesky or Doolittle) based on matrix properties.
    """
    # Matrix System 1
    problem1 = [
        [1, -1, 3, 2, 15],
        [-1, 5, -5, -2, -35],
        [3, -5, 19, 3, 94],
        [2, -2, 3, 21, 1]
    ]

    # Matrix System 2
    problem2 = [
        [4, 2, 4, 0, 20],
        [2, 2, 3, 2, 36],
        [4, 3, 6, 3, 60],
        [0, 2, 3, 9, 122]
    ]

    print("\nSolving System 1:")
    x1 = solve_system(problem1)
    print("Solution:", [round(float(i), 3) for i in x1])  # Fixed Output

    print("\nSolving System 2:")
    x2 = solve_system(problem2)
    print("Solution:", [round(float(i), 3) for i in x2])  # Fixed Output


if __name__ == "__main__":
    main()
