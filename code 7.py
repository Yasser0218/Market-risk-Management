# Python code to demonstrate the difference between Bootstrapping and Monte Carlo Simulation

import numpy as np 
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Observed data (e.g., daily returns of a stock)
data = np.random.normal(loc=0.05, scale=0.2, size=100)  # Mean = 0.05, Std Dev = 0.2, 100 samples

# BOOTSTRAPPING
# Resampling with replacement
num_bootstrap_samples = 1000
bootstrap_means = []
for _ in range(num_bootstrap_samples):
    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    bootstrap_means.append(np.mean(bootstrap_sample))

# MONTE CARLO SIMULATION
# Simulating from a known distribution (Normal with the same mean and std as the data)
data_mean = np.mean(data)
data_std = np.std(data)
num_monte_carlo_samples = 1000
monte_carlo_samples = np.random.normal(loc=data_mean, scale=data_std, size=(num_monte_carlo_samples, len(data)))
monte_carlo_means = np.mean(monte_carlo_samples, axis=1)

# PLOT COMPARISON
plt.figure(figsize=(12, 6))

# Bootstrapping distribution
plt.subplot(1, 2, 1)
plt.hist(bootstrap_means, bins=30, color='skyblue', edgecolor='black')
plt.title('Bootstrapping: Distribution of Means')
plt.xlabel('Mean')
plt.ylabel('Frequency')

# Monte Carlo distribution
plt.subplot(1, 2, 2)
plt.hist(monte_carlo_means, bins=30, color='salmon', edgecolor='black')
plt.title('Monte Carlo: Distribution of Means')
plt.xlabel('Mean')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Summary statistics
print("Bootstrapping Results:")
print(f"Mean of bootstrap means: {np.mean(bootstrap_means):.4f}")
print(f"Standard deviation of bootstrap means: {np.std(bootstrap_means):.4f}")
print("\nMonte Carlo Results:")
print(f"Mean of Monte Carlo means: {np.mean(monte_carlo_means):.4f}")
print(f"Standard deviation of Monte Carlo means: {np.std(monte_carlo_means):.4f}")