# region imports
from math import gamma, sqrt, pi
from numericalMethods import Simpson


# endregion

# region function definitions
def Km(m):
    """
    Computes the constant Km for the t-distribution.
    """
    return gamma((m + 1) / 2) / (sqrt(m * pi) * gamma(m / 2))


def t_distribution_CDF(z, m):
    """
    Computes the cumulative distribution function (CDF) of the t-distribution.
    """
    integrand = lambda u: (1 + (u ** 2) / m) ** (-(m + 1) / 2)
    return Km(m) * Simpson(integrand, (-10, z), N=1000)  # Integrate from -∞ (approximated as -10) to z


def main():
    """
    Main function to compute the CDF of the t-distribution for user-defined degrees of freedom and z-values.
    """
    print("This program calculates the t-distribution CDF and compares it to Table A9.")

    degrees_of_freedom = int(input("Enter degrees of freedom (7, 11, or 15): "))
    if degrees_of_freedom not in [7, 11, 15]:
        print("Invalid input. Please enter 7, 11, or 15.")
        return

    z_values = []
    for i in range(3):
        z = float(input(f"Enter z-value {i + 1}: "))
        z_values.append(z)

    print("\nResults:")
    for z in z_values:
        probability = t_distribution_CDF(z, degrees_of_freedom)
        print(f"F({z:.3f}) with df={degrees_of_freedom}: {probability:.5f}")


# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion
