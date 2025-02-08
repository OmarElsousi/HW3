# region imports
from copy import deepcopy as dcpy
from math import cos, pi
import numericalMethods as nm
import matrixOperations as mo


# endregion

# region Functions
def LUFactorization(A):
    """
    Performs LU decomposition using Doolittle's method with partial pivoting.
    A = L * U where L is lower triangular with ones on the diagonal, and U is upper triangular.
    :param A: a nxn matrix
    :return: (L, U) where L and U are lower and upper triangular matrices.
    """
    A = dcpy(A)  # Ensure we don't modify the original matrix
    n = len(A)
    U = [[0] * n for _ in range(n)]
    L = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    # Apply partial pivoting
    for i in range(n):
        max_row = max(range(i, n), key=lambda k: abs(A[k][i]))  # Find row with max absolute value
        if i != max_row:  # Swap rows in A
            A[i], A[max_row] = A[max_row], A[i]

        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))

        for j in range(i + 1, n):
            if U[i][i] == 0:
                raise ValueError("Matrix is singular or near-singular.")
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U


def BackSolve(A, b, UT=True):
    """
    This is a backsolving algorithm for a matrix and b vector where A is triangular.
    :param A: A triangularized matrix (Upper or Lower).
    :param b: The right-hand side of a matrix equation Ax=b.
    :param UT: Boolean - True for upper triangular, False for lower triangular.
    :return: The solution vector x.
    """
    # Ensure b is a flat list (not a column vector)
    if isinstance(b[0], list):
        b = [row[0] for row in b]

    nRows = len(b)
    x = [0] * nRows

    if UT:
        for nR in range(nRows - 1, -1, -1):
            s = sum(A[nR][nC] * x[nC] for nC in range(nR + 1, nRows))
            x[nR] = (b[nR] - s) / A[nR][nR]
    else:
        for nR in range(nRows):
            s = sum(A[nR][nC] * x[nC] for nC in range(nR))
            x[nR] = (b[nR] - s) / A[nR][nR]

    return x


def Doolittle(Aaug):
    """
    The Doolittle method for solving the matrix equation [A][x]=[b]:
    Step 1:  Factor [A]=[L][U]
    Step 2:  Solve [L][y]=[b] for [y]
    Step 3:  Solve [U][x]=[y] for [x]
    :param Aaug: The augmented matrix.
    :return: The solution vector x.
    """
    A, b = mo.separateAugmented(Aaug)
    L, U = LUFactorization(A)
    y = BackSolve(L, b, UT=False)
    x = BackSolve(U, y, UT=True)
    return x  # x should be a 1D list


def main():
    # Define example matrices
    A = [[3, 5, 2], [0, 8, 2], [6, 2, 8]]
    L, U = LUFactorization(A)

    print("L:")
    for r in L:
        print(r)

    print("\nU:")
    for r in U:
        print(r)

    aug = [
        [3, 1, -1, 2],
        [1, 4, 1, 12],
        [2, 1, 2, 10]
    ]

    x = Doolittle(aug)
    x = [round(y, 3) for y in x]
    print("x:", x)

    y = nm.GaussSeidel(aug, [0, 0, 0])
    y = [round(z, 3) for z in y]
    print("Gauss-Seidel solution:", y)


# endregion

if __name__ == "__main__":
    main()
