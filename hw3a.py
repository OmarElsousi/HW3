# region imports
from numericalMethods import GPDF, Probability, Secant


# endregion

# region function definitions
def probability_error(c, target_P, mean, stDev, GT, OneSided):
    """
    Computes the difference between the actual probability and the target probability.
    Used as the function for the Secant method to find the critical value c.

    Parameters:
        c (float): The critical value for which probability is being computed.
        target_P (float): The target probability.
        mean (float): The mean of the distribution.
        stDev (float): The standard deviation of the distribution.
        GT (bool): If True, computes P(x > c), otherwise P(x < c).
        OneSided (bool): If True, computes one-sided probability, otherwise two-sided.

    Returns:
        float: The difference between the computed probability and the target probability.
    """
    if OneSided:
        prob = Probability(GPDF, (mean, stDev), c, GT=GT)
    else:
        prob = Probability(GPDF, (mean, stDev), c, GT=True)
        prob = 1 - 2 * prob
    return prob - target_P


def main():
    """
    Interactive program to compute probabilities for normal distributions.
    It allows the user to:
    1. Compute P(x < c) or P(x > c) for a given c.
    2. Compute c given P(x < c) or P(x > c) using the Secant method.
    """
    Again = True
    yesOptions = ["y", "yes", "true"]
    while Again:
        response = input(
            "Are you specifying c and solving for P, or specifying P and solving for c? (Enter 'c' or 'p'): "
        ).strip().lower()
        solveForC = response == "p"

        mean = float(input("Population mean? (default 0): ") or 0)
        stDev = float(input("Standard deviation? (default 1): ") or 1)

        response = input("One-sided probability (default: True)? (y/n): ").strip().lower()
        OneSided = response in yesOptions

        response = input("Probability greater than c? (default: False)? (y/n): ").strip().lower()
        GT = response in yesOptions

        if solveForC:
            target_P = float(input("Enter the probability P: "))
            c0, c1 = mean, mean + stDev  # Initial guesses for Secant method
            c, _ = Secant(lambda c: probability_error(c, target_P, mean, stDev, GT, OneSided), c0, c1)
            print(f"The value of c that gives probability {target_P:.3f} is {c:.3f}")
        else:
            c = float(input("Enter the value of c: "))
            if OneSided:
                prob = Probability(GPDF, (mean, stDev), c, GT=GT)
                print(f"P(x {'>' if GT else '<'} {c:.3f} | {mean:.3f}, {stDev:.3f}) = {prob:.3f}")
            else:
                prob = Probability(GPDF, (mean, stDev), c, GT=True)
                prob = 1 - 2 * prob
                print(
                    f"P({mean - (c - mean):.3f} < x < {mean + (c - mean):.3f} | {mean:.3f}, {stDev:.3f}) = {prob:.3f}"
                )

        response = input("Go again? (y/n): ").strip().lower()
        Again = response in yesOptions


# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion
