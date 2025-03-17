import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy.stats import johnsonsu, norm

class EmpiricalCorrelationAnalysis:
    def __init__(self, start_date='1972-01-01', end_date='2017-07-31'):
        self.start_date = start_date
        self.end_date = end_date
        np.random.seed(42)

    def generate_sample_data(self, n_stocks=30, n_days=11214):
        """
        Generate sample data similar to the Dow study
        """
        # Generate daily returns
        returns = pd.DataFrame(
            np.random.normal(0.0001, 0.02, (n_days, n_stocks)),
            columns=[f'Stock_{i+1}' for i in range(n_stocks)]
        )

        # Generate economic states (Expansion, Normal, Recession)
        states = np.random.choice(
            ['Expansion', 'Normal', 'Recession'],
            size=n_days,
            p=[0.3, 0.5, 0.2]
        )

        return returns, states

class EconomicStateAnalysis:
      def analyze_by_state(self, returns, states):
          """
          Analyze correlation patterns in different economic states
          """
          print("\n1. Correlation Analysis by Economic State")
          print("-" * 50)

          results = {}
          # Ensure states is a Series with the same index as returns
          states = pd.Series(states, index=returns.index)

          for state in ['Expansion', 'Normal', 'Recession']:
              # Filter by state
              mask = states == state
              state_returns = returns[mask]

              if not state_returns.empty:
                  # Calculate correlation matrix
                  corr_matrix = state_returns.corr()

                  # Get average correlation (excluding diagonal)
                  mask = np.ones_like(corr_matrix, dtype=bool)
                  np.fill_diagonal(mask, False)
                  avg_corr = corr_matrix[mask].mean().mean()

                  # Calculate correlation volatility
                  rolling_corr = state_returns.rolling(window=22).corr()
                  corr_vol = rolling_corr.groupby(level=0).std()
                  avg_corr_vol = corr_vol.mean().mean()

                  results[state] = {
                      'avg_correlation': avg_corr,
                      'correlation_volatility': avg_corr_vol
                  }

                  print(f"\n{state}:")
                  print(f"Average Correlation: {avg_corr:.4f}")
                  print(f"Correlation Volatility: {avg_corr_vol:.4f}")
              else:
                  print(f"\n{state}: No data available.")

          return results


class MeanReversionAnalysis:
      def test_mean_reversion(self, correlation_series):
          """
          Test for mean reversion in correlation series
          """
          print("\n2. Mean Reversion Analysis")
          print("-" * 50)

          # Calculate changes and lagged levels
          changes = correlation_series.diff().dropna().reset_index(drop=True)
          lagged_levels = correlation_series.shift(1).dropna().reset_index(drop=True)

          # Align lengths
          min_length = min(len(changes), len(lagged_levels))
          changes = changes.iloc[:min_length]
          lagged_levels = lagged_levels.iloc[:min_length]

          # Check for valid data
          if len(changes) == 0 or len(lagged_levels) == 0:
              print("Error: Insufficient data for regression. Ensure non-empty inputs.")
              return None, None

          # Run regression
          X = sm.add_constant(lagged_levels, has_constant='add')
          model = OLS(changes, X).fit()

          # Print model parameters for debugging
          print("Regression Parameters:")
          print(model.params)

          # Access the parameter by name
          mean_reversion_rate = -model.params.get("x1", float('nan'))

          print(f"Mean Reversion Rate: {mean_reversion_rate:.4f}")

          # Visualization: Scatter Plot
          plt.figure(figsize=(8, 6))
          plt.scatter(lagged_levels, changes, alpha=0.5)
          plt.plot(lagged_levels, model.predict(X), color='red', label='Fitted Line')
          plt.title('Mean Reversion Analysis')
          plt.xlabel('Lagged Levels')
          plt.ylabel('Changes')
          plt.legend()
          plt.grid(True)
          plt.show()

          return mean_reversion_rate, model




class AutocorrelationAnalysis:
    def analyze_autocorrelation(self, correlation_series, max_lags=9):
        """
        Analyze autocorrelation patterns
        """
        print("\n3. Autocorrelation Analysis")
        print("-" * 50)

        autocorr = [correlation_series.autocorr(lag=i) for i in range(1, max_lags + 1)]

        # Visualization: Autocorrelation Bar Plot
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, max_lags + 1), autocorr, color='blue', alpha=0.7)
        plt.title('Autocorrelation by Lag')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True)
        plt.show()

        for i, ac in enumerate(autocorr, 1):
            print(f"Lag {i}: {ac:.4f}")

        return autocorr

class DistributionAnalysis:
    def analyze_distributions(self, correlation_series):
        """
        Analyze correlation distributions and fit distributions
        """
        print("\n4. Distribution Analysis")
        print("-" * 50)

        # Histogram and Fits
        plt.figure(figsize=(10, 6))
        sns.histplot(correlation_series, bins=30, kde=True, stat='density', color='skyblue')

        # Fit distributions
        x = np.linspace(correlation_series.min(), correlation_series.max(), 100)
        params_norm = norm.fit(correlation_series)
        pdf_norm = norm.pdf(x, *params_norm)

        plt.plot(x, pdf_norm, 'r-', label='Normal Fit')
        plt.title('Distribution Fit')
        plt.xlabel('Correlation')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Normal Fit Parameters: Mean = {params_norm[0]:.4f}, Std = {params_norm[1]:.4f}")

        return params_norm

def main():
    # Generate sample data
    eca = EmpiricalCorrelationAnalysis()
    returns, states = eca.generate_sample_data()

    # Economic State Analysis
    esa = EconomicStateAnalysis()
    state_results = esa.analyze_by_state(returns, states)

    # Calculate rolling correlations
    rolling_corr = returns.iloc[:, 0].rolling(window=22).corr(returns.iloc[:, 1]).dropna()

    # Mean Reversion Analysis
    mra = MeanReversionAnalysis()
    mean_reversion_rate, model = mra.test_mean_reversion(rolling_corr)

    # Autocorrelation Analysis
    aca = AutocorrelationAnalysis()
    autocorr = aca.analyze_autocorrelation(rolling_corr)

    # Distribution Analysis
    da = DistributionAnalysis()
    dist_params = da.analyze_distributions(rolling_corr)

if __name__ == "__main__":
    main()
