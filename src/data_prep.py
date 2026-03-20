import numpy as np
import pandas as pd
import os


def generate_data(n=1000, random_state=42):
    tenure = np.random.exponential(scale = 24, size=n).clip(1,72).astype(int)
    monthly_charges = np.random.normal(loc = 65, scale=20, size=n).clip(10, 150).round(2)
    churn_prob = 1/(np.exp(0.05 * tenure - 0.02 * monthly_charges))
    y = (np.random.rand(n) < churn_prob).astype(int)
    df = pd.DataFrame({'tenure': tenure, 'monthly_charges':monthly_charges, 'churn': y})
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_data()
    df.to_csv("data/churn.csv", index=False)
    
