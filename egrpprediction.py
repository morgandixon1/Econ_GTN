import math
import pandas as pd

# Hard-coded data for existing businesses
data = {
    "Name of business": ["Dans Auto Shop", "Patricias Bakery", "Edwards Electric", "Verizon", "Patties Pet Shop", "Jimmy Johns", "Anti Corp Company"],
    "Prediction Score 0-1": [0.3, 0, 0.3, 0.6, 0, 0, 0],
    "Type of Business": ["Automotive Mechanic", "Bakery", "Electrician", "Phone Company", "Pet Shop", "Food", "Non-profit"],
    "Economic Gross Regional Product": [25560000, 400000, 769600, 1240000, 240000, 657000, 8000]
}

df = pd.DataFrame(data)

# Function to estimate the EGRP for a new business
def estimate_egrp(new_business_demand, num_employees):
    avg_shared_revenue = sum(df['Prediction Score 0-1'] * df['Economic Gross Regional Product']) / len(df)
    log_term = math.log(2 + num_employees)  # Increase the base of the logarithm for a stronger effect
    exp_term = math.exp(0.01 * new_business_demand * num_employees)  # Introduce exponential term
    estimated_egrp = (new_business_demand * 1.5) * (1 + log_term) * avg_shared_revenue * exp_term  # Multiply demand by a factor to increase its impact

    return f"The estimated EGRP for the new business is approximately ${estimated_egrp:.2f}."

# Entry form for input data
print("For the Demand: Enter a number between 0 and 1 to represent the percentage of the local population that is expected to frequent the new business more than once a year.")
new_business_demand = float(input("Enter the estimated demand for the new business (0-1): "))

print("For the Number of Employees: Enter the estimated number of employees for the new business.")
num_employees = int(input("Enter the estimated number of employees: "))

# Run the function to estimate the EGRP for the new business
print(estimate_egrp(new_business_demand, num_employees))
