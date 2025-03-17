import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Fetch Nvidia daily price data for the last three year
Nvidia = yf.download('NVDA', start='2021-01-01', end='2024-01-01')

# Calculate daily returns
Nvidia['Daily Return'] = Nvidia['Adj Close'].pct_change().dropna()
returns = Nvidia['Daily Return'].dropna()

# VaR and ES Parameters
confidence_level = 0.95
alpha = 1 - confidence_level

# 1. Historical VaR
historical_var = np.percentile(returns, 100 * alpha)

# 2. Parametric (Normal) VaR
mean_return = np.mean(returns)
std_dev = np.std(returns)
normal_var = mean_return + std_dev * np.percentile(np.random.normal(0, 1, len(returns)), 100 * alpha)

# 3. Parametric (Lognormal) VaR
# Assumption: Returns are lognormally distributed
log_returns = np.log1p(returns)
log_mean = np.mean(log_returns)
log_std = np.std(log_returns)
lognormal_var = np.exp(log_mean + log_std * np.percentile(np.random.normal(0, 1, len(returns)), 100 * alpha)) - 1

# 4. Expected Shortfall (ES)
def expected_shortfall(returns, confidence_level):
    var = np.percentile(returns, 100 * (1 - confidence_level))
    es = returns[returns <= var].mean()
    return es

es = expected_shortfall(returns, confidence_level)

# Print results
print("Value at Risk (VaR) and Expected Shortfall (ES):")
print(f"Historical VaR (95%): {historical_var:.6f}")
print(f"Parametric Normal VaR (95%): {normal_var:.6f}")
print(f"Parametric Lognormal VaR (95%): {lognormal_var:.6f}")
print(f"Expected Shortfall (ES) (95%): {es:.6f}")

# Plot VaR and ES
plt.figure(figsize=(10, 6))
plt.hist(returns, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(historical_var, color='red', linestyle='--', label=f'Historical VaR (95%): {historical_var:.4f}')
plt.axvline(es, color='green', linestyle='--', label=f'Expected Shortfall (ES): {es:.4f}')
plt.title('Bitcoin Daily Returns and Risk Metrics')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.legend()
plt.show()

