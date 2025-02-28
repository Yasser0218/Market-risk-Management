#Calculating Undiversified and Diversified VaR
import numpy as np

def calculate_undiversified_var(*args):
    """
    Calculate undiversified VaR by summing individual VaRs.
    Input: Individual VaRs (args)
    """
    return sum(args)

def calculate_diversified_var(vars, correlations):
    """
    Calculate diversified VaR using variances and correlations.
    Input:
        vars: list of individual VaRs
        correlations: correlation matrix
    """
    # Ensure correlation matrix is numpy array
    correlations = np.array(correlations)
    # Convert VaRs to variances (square them)
    variances = np.square(vars)
    # Calculate portfolio variance
    portfolio_variance = np.dot(variances, correlations.dot(variances))
    return np.sqrt(portfolio_variance)
