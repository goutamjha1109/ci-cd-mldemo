import numpy as np
import pandas as pd
import os
from utils import get_args, load_config
from logger import logger

def generate_data(n=1000, random_state=42):
    logger.info(f"Generating data with samples {n}")
    tenure = np.random.exponential(scale = 24, size=n).clip(1,72).astype(int)
    monthly_charges = np.random.normal(loc = 65, scale=20, size=n).clip(10, 150).round(2)
    churn_prob = 1/(np.exp(0.05 * tenure - 0.02 * monthly_charges))
    y = (np.random.rand(n) < churn_prob).astype(int)
    df = pd.DataFrame({'tenure': tenure, 'monthly_charges':monthly_charges, 'churn': y})
    logger.info(f"Generated data of shape {df.shape}")
    return df


if __name__ == "__main__":
    args = get_args()
    params = load_config(args.config)

    data_params = params['data']
    os.makedirs("data", exist_ok=True)
    df = generate_data(n=data_params["n_samples"], random_state=data_params["random_state"])
    df.to_csv("data/churn.csv", index=False)
    
