import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class CorrelationAnalysis:
    def __init__(self):
        np.random.seed(42)
        # Example data from Chapter 7
        self.sample_data = {
            'Year': [2014, 2015, 2016, 2017, 2018],
            'Asset_X': [20.00, -10.00, 75.93, -15.79, 75.00],
            'Asset_Y': [15.00, 100.00, -10.87, 17.07, -20.83]
        }

    def basic_correlation_example(self):
        """
        Demonstrates basic correlation calculation using Chapter 7's example
        """
        print("\n1. Basic Correlation Analysis")
        print("-" * 50)

        df = pd.DataFrame(self.sample_data)
        correlation = df['Asset_X'].corr(df['Asset_Y'])

        # Calculate statistics
        stats_x = df['Asset_X'].agg(['mean', 'std']).round(2)
        stats_y = df['Asset_Y'].agg(['mean', 'std']).round(2)

        print("\nSample Returns Data:")
        print(df)
        print(f"\nCorrelation between assets: {correlation:.4f}")
        print(f"Asset X - Mean: {stats_x['mean']}%, Std: {stats_x['std']}%")
        print(f"Asset Y - Mean: {stats_y['mean']}%, Std: {stats_y['std']}%")

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Asset_X'], df['Asset_Y'])
        plt.title('Asset X vs Asset Y Returns')
        plt.xlabel('Asset X Returns (%)')
        plt.ylabel('Asset Y Returns (%)')
        plt.grid(True)
        plt.show()

        return df

class PortfolioAnalysis:
    def __init__(self, investment_x=10000000, investment_y=5000000):
        self.investment_x = investment_x  # $10 million
        self.investment_y = investment_y  # $5 million

    def calculate_var(self, sigma_x=0.02, sigma_y=0.01, correlation=0.7,
                     confidence_level=0.99, time_horizon=10):
        """
        Calculates VaR based on Example 7.2 from the chapter
        """
        print("\n2. Value at Risk (VaR) Calculation")
        print("-" * 50)

        # Calculate covariances
        cov_xx = sigma_x * sigma_x
        cov_xy = correlation * sigma_x * sigma_y
        cov_yy = sigma_y * sigma_y

        # Create covariance matrix
        cov_matrix = np.array([[cov_xx, cov_xy],
                              [cov_xy, cov_yy]])

        # Create position vector
        positions = np.array([self.investment_x, self.investment_y])

        # Calculate portfolio standard deviation
        portfolio_variance = positions.T @ cov_matrix @ positions
        portfolio_std = np.sqrt(portfolio_variance)

        # Calculate VaR
        z_score = stats.norm.ppf(confidence_level)
        var = portfolio_std * z_score * np.sqrt(time_horizon)

        print(f"Portfolio Standard Deviation: {portfolio_std:,.2f}")
        print(f"10-day VaR at 99% confidence: ${var:,.2f}")

        return var

    def correlation_impact_on_var(self, sigma_x=0.02, sigma_y=0.01,
                                confidence_level=0.99, time_horizon=10):
        """
        Demonstrates how correlation affects VaR (Figure 7.6 concept)
        """
        print("\n3. Correlation Impact on VaR")
        print("-" * 50)

        correlations = np.linspace(-1, 1, 21)
        vars = []

        for corr in correlations:
            var = self.calculate_var(sigma_x, sigma_y, corr, confidence_level, time_horizon)
            vars.append(var)

        plt.figure(figsize=(10, 6))
        plt.plot(correlations, vars)
        plt.title('VaR vs Correlation')
        plt.xlabel('Correlation')
        plt.ylabel('VaR ($)')
        plt.grid(True)
        plt.show()

        return pd.DataFrame({'Correlation': correlations, 'VaR': vars})

class CorrelationSwap:
    def __init__(self, notional=1000000):
        self.notional = notional

    def calculate_realized_correlation(self, correlation_matrix):
        """
        Calculates realized correlation based on Example 7.1
        """
        print("\n4. Correlation Swap Analysis")
        print("-" * 50)

        n = len(correlation_matrix)
        sum_correlations = 0
        count = 0

        # Sum upper triangle correlations (excluding diagonal)
        for i in range(n):
            for j in range(i+1, n):
                sum_correlations += correlation_matrix[i, j]
                count += 1

        realized_correlation = (2 / (n * (n-1))) * sum_correlations
        return realized_correlation

    def swap_payoff(self, realized_correlation, fixed_correlation=0.10):
        """
        Calculates correlation swap payoff
        """
        payoff = self.notional * (realized_correlation - fixed_correlation)

        print(f"Realized Correlation: {realized_correlation:.4%}")
        print(f"Fixed Correlation: {fixed_correlation:.4%}")
        print(f"Swap Payoff: ${payoff:,.2f}")

        return payoff

def main():
    # Basic correlation analysis
    ca = CorrelationAnalysis()
    df = ca.basic_correlation_example()

    # Portfolio VaR analysis
    pa = PortfolioAnalysis()
    var = pa.calculate_var()
    var_analysis = pa.correlation_impact_on_var()

    # Correlation swap example
    cs = CorrelationSwap()
    # Example correlation matrix from the chapter
    corr_matrix = np.array([[1.0, 0.5, 0.1],
                           [0.5, 1.0, 0.3],
                           [0.1, 0.3, 1.0]])

    realized_corr = cs.calculate_realized_correlation(corr_matrix)
    payoff = cs.swap_payoff(realized_corr)

if __name__ == "__main__":
    main()