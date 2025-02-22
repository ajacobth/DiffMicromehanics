import pandas as pd
import numpy as np

# Generate two uniform random variables with 8192 samples
data = {
    'Var1': np.random.uniform(0, 1, 8192),
    'Var2': np.random.uniform(0, 1, 8192)
}

# Create DataFrame from generated data
df = pd.DataFrame(data)

# Define third-order polynomial function
def third_order_polynomial(var1, var2):
    return var1**3 + var2**3 + 2*var1*var2 + 3*var1**2*var2 + var1*var2**2

# Apply polynomial transformation
df['Output'] = third_order_polynomial(df['Var1'], df['Var2'])

# Save as CSV with specified column order
df.to_csv('two_variable_polynomial_dataset.csv', index=False)

