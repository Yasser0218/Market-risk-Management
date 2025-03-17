import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

class VaRMapper:
    def __init__(self):
        """Initialize VaR mapping calculator"""
        self.confidence_level = 0.95
        self.z_score = norm.ppf(self.confidence_level)

    # Section 1: Currency Forward Contract VaR
    def calculate_forward_contract_var(self, spot_price, forward_price, notional,
                                       spot_vol, r_domestic, r_foreign,
                                       bill_vol_domestic, bill_vol_foreign,
                                       correlations, time_to_maturity):
        """
        Calculate VaR for currency forward contract
        """
        # Calculate discount factors
        df_domestic = np.exp(-r_domestic * time_to_maturity)
        df_foreign = np.exp(-r_foreign * time_to_maturity)

        # Calculate present values
        spot_exposure = notional * spot_price * df_foreign
        foreign_bill_exposure = spot_exposure
        domestic_bill_exposure = -notional * forward_price * df_domestic

        # Calculate individual VaRs
        spot_var = abs(spot_exposure * spot_vol)
        foreign_bill_var = abs(foreign_bill_exposure * bill_vol_foreign)
        domestic_bill_var = abs(domestic_bill_exposure * bill_vol_domestic)

        # Calculate undiversified VaR
        undiversified_var = spot_var + foreign_bill_var + domestic_bill_var

        # Calculate diversified VaR using correlation matrix
        variance = (spot_var**2 +
                   foreign_bill_var**2 +
                   domestic_bill_var**2 +
                   2 * spot_var * foreign_bill_var * correlations['spot_foreign'] +
                   2 * spot_var * domestic_bill_var * correlations['spot_domestic'] +
                   2 * foreign_bill_var * domestic_bill_var * correlations['foreign_domestic'])

        diversified_var = np.sqrt(variance)

        return {
            'spot_exposure': spot_exposure,
            'foreign_bill_exposure': foreign_bill_exposure,
            'domestic_bill_exposure': domestic_bill_exposure,
            'spot_var': spot_var,
            'foreign_bill_var': foreign_bill_var,
            'domestic_bill_var': domestic_bill_var,
            'undiversified_var': undiversified_var,
            'diversified_var': diversified_var
        }

    # Section 2: Commodity Forward Contract VaR
    def calculate_commodity_forward_var(self, price, quantity, vol, time_to_maturity, r):
        """
        Calculate VaR for commodity forward contract
        """
        present_value = quantity * price * np.exp(-r * time_to_maturity)
        var = present_value * vol

        return {
            'present_value': present_value,
            'var': var
        }

    # Section 3: Forward Rate Agreement (FRA) VaR
    def calculate_fra_var(self, notional, short_rate, long_rate,
                          short_vol, long_vol, correlation,
                          time_to_short, time_to_long):
        """
        Calculate VaR for Forward Rate Agreement
        """
        # Calculate present values
        df_short = 1 / (1 + short_rate/200)  # Using discrete compounding
        present_value = notional * df_short

        # Calculate individual VaRs
        short_var = abs(present_value * short_vol)
        long_var = abs(present_value * long_vol)

        # Calculate undiversified VaR
        undiversified_var = short_var + long_var

        # Calculate diversified VaR
        variance = short_var**2 + long_var**2 + 2 * correlation * short_var * long_var
        diversified_var = np.sqrt(variance)

        return {
            'present_value': present_value,
            'short_var': short_var,
            'long_var': long_var,
            'undiversified_var': undiversified_var,
            'diversified_var': diversified_var
        }

# Example usage
# Main function to demonstrate VaR mapping

def main():
    var_mapper = VaRMapper()

    # Example 1: Currency Forward (EUR/USD)
    print("1. EUR/USD Forward Contract Example")
    forward_results = var_mapper.calculate_forward_contract_var(
        spot_price=1.2877,
        forward_price=1.3009,
        notional=100_000_000,  # EUR 100M
        spot_vol=0.045381,
        r_domestic=0.033304,
        r_foreign=0.022810,
        bill_vol_domestic=0.002121,
        bill_vol_foreign=0.001396,
        correlations={
            'spot_foreign': 0.1289,
            'spot_domestic': 0.0400,
            'foreign_domestic': -0.0583
        },
        time_to_maturity=1.0
    )
    print(f"Diversified VaR: ${forward_results['diversified_var']:,.0f}")
    print(f"Undiversified VaR: ${forward_results['undiversified_var']:,.0f}")

    # Check and handle negative exposure values
    exposures = [forward_results['spot_exposure'],
                forward_results['foreign_bill_exposure'],
                forward_results['domestic_bill_exposure']]
    exposures = [abs(x) for x in exposures]  # Convert negative values to positive for pie chart

    # Visualization: Pie Chart for Exposures
    labels = ['Spot Exposure', 'Foreign Bill Exposure', 'Domestic Bill Exposure']
    plt.figure(figsize=(6, 6))
    plt.pie(exposures, labels=labels, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Exposure Contributions in Currency Forward Contract')
    plt.show()

    # Example 2: Oil Forward Contract
    print("\n2. Oil Forward Contract Example")
    oil_results = var_mapper.calculate_commodity_forward_var(
        price=45.2,
        quantity=1_000_000,  # 1M barrels
        vol=0.1405,  # 14.05%
        time_to_maturity=1.0,
        r=0.033304
    )
    print(f"Present Value: ${oil_results['present_value']:,.0f}")
    print(f"VaR: ${oil_results['var']:,.0f}")

    # Visualization: Stacked Bar Chart for Present Value and VaR
    plt.figure(figsize=(8, 5))
    plt.bar(['Oil Forward'], [oil_results['present_value']], color='orange', label='Present Value')
    plt.bar(['Oil Forward'], [oil_results['var']], color='red', bottom=[oil_results['present_value']], label='VaR')
    plt.title('Oil Forward Contract: Present Value and VaR')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.show()

    # Example 3: FRA
    print("\n3. Forward Rate Agreement Example")
    fra_results = var_mapper.calculate_fra_var(
        notional=100_000_000,  # USD 100M
        short_rate=0.056250,   # 5.6250%
        long_rate=0.058125,    # 5.8125%
        short_vol=0.001629,
        long_vol=0.004696,
        correlation=0.8738,
        time_to_short=0.5,
        time_to_long=1.0
    )
    print(f"Diversified VaR: ${fra_results['diversified_var']:,.0f}")
    print(f"Undiversified VaR: ${fra_results['undiversified_var']:,.0f}")

    # Visualization: Histogram with VaR Thresholds
    fra_components = np.random.normal(loc=fra_results['diversified_var'], scale=fra_results['diversified_var']*0.1, size=1000)
    plt.figure(figsize=(8, 5))
    plt.hist(fra_components, bins=30, alpha=0.6, color='purple', label='Simulated Distribution')
    plt.axvline(fra_results['diversified_var'], color='green', linestyle='--', label=f'Diversified VaR: ${fra_results["diversified_var"]:,.0f}')
    plt.axvline(fra_results['undiversified_var'], color='red', linestyle='--', label=f'Undiversified VaR: ${fra_results["undiversified_var"]:,.0f}')
    plt.title('FRA: Diversified vs Undiversified VaR')
    plt.xlabel('Value ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()