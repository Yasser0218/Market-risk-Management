# Python code to demonstrate when to use Bootstrapping vs Monte Carlo Simulation

import numpy as np
import matplotlib.pyplot as plt

# 1. BOOTSTRAPPING EXAMPLE
# Dataset: Observed housing prices (skewed distribution)
housing_prices = np.random.lognormal(mean=12, sigma=0.5, size=100)  # Simulated skewed data

# Bootstrapping to estimate 95% confidence interval for the median
num_bootstrap_samples = 1000
bootstrap_medians = []
for _ in range(num_bootstrap_samples):
    bootstrap_sample = np.random.choice(housing_prices, size=len(housing_prices), replace=True)
    bootstrap_medians.append(np.median(bootstrap_sample))

# Calculate confidence interval
lower_ci = np.percentile(bootstrap_medians, 2.5)
upper_ci = np.percentile(bootstrap_medians, 97.5)

print("Bootstrapping Example:")
print(f"Observed median: {np.median(housing_prices):.2f}")
print(f"95% Confidence Interval for median: [{lower_ci:.2f}, {upper_ci:.2f}]\n")

# Plot Bootstrapping Results
plt.figure(figsize=(10, 5))
plt.hist(bootstrap_medians, bins=30, color='skyblue', edgecolor='black')
plt.axvline(lower_ci, color='red', linestyle='--', label=f'Lower CI: {lower_ci:.2f}')
plt.axvline(upper_ci, color='green', linestyle='--', label=f'Upper CI: {upper_ci:.2f}')
plt.title('Bootstrapping: Median Confidence Interval for Observed housing prices')
plt.xlabel('Median')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# 2. MONTE CARLO SIMULATION EXAMPLE
# Portfolio risk modeling assuming normal distribution of returns
portfolio_mean_return = 0.08  # Assumed annual return
portfolio_std_dev = 0.15     # Assumed standard deviation
num_years = 10
num_simulations = 1000

# Simulating portfolio returns over 10 years
simulated_returns = []
for _ in range(num_simulations):
    annual_returns = np.random.normal(loc=portfolio_mean_return, scale=portfolio_std_dev, size=num_years)
    total_return = np.prod(1 + annual_returns) - 1  # Compound return
    simulated_returns.append(total_return)

# Calculate risk metrics
mean_return = np.mean(simulated_returns)
var_95 = np.percentile(simulated_returns, 5)  # 5% Value at Risk

print("Monte Carlo Simulation Example:")
print(f"Mean portfolio return over 10 years: {mean_return:.2f}")
print(f"95% Value at Risk (VaR): {var_95:.2f}\n")

# Plot Monte Carlo Results
plt.figure(figsize=(10, 5))
plt.hist(simulated_returns, bins=30, color='salmon', edgecolor='black')
plt.axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.2f}')
plt.title('Monte Carlo Simulation: Portfolio Returns over 10 years')
plt.xlabel('Total Return')
plt.ylabel('Frequency')
plt.legend()
plt.show()
