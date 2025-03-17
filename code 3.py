# Example 1: Currency Forward Contract
print("Example 1: Currency Forward Contract")
spot_var = 20000
foreign_bill_var = 15000
domestic_bill_var = 18000
correlations_1 = [
    [1.0, 0.2, 0.1],
    [0.2, 1.0, -0.1],
    [0.1, -0.1, 1.0]
]


# Example 2: Commodity Forward Contract
print("\nExample 2: Commodity Forward Contract")
oil_var = 30000
transport_var = 12000
inventory_var = 10000
correlations_2 = [
    [1.0, 0.3, 0.2],
    [0.3, 1.0, 0.4],
    [0.2, 0.4, 1.0]
]


# Example 3: FRA
print("\nExample 3: Forward Rate Agreement")
short_rate_var = 25000
long_rate_var = 20000
correlations_3 = [
    [1.0, 0.5],
    [0.5, 1.0]
]
