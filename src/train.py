import yaml
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os
import json 
import pickle
from logger import logger
from data_prep import generate_data
from pathlib import Path
import pandas as pd
from utils import get_args,load_config

def load_data(data_params):
    df = pd.DataFrame()
    try:
        logger.info("Loading data............")
        path = Path(data_params["path"])
        df = pd.read_csv(path)
        logger.info("Data loading successful.............")
    except Exception as e:  
        logger.error(f"Error loading dataframe {e}")
    return df

# def load_config(args):
#     with open(args.config, "r") as f:
#         params = yaml.safe_load(f)
#     return params["model"], params["data"]

def split_data(df, params):
    y = df["churn"]
    X = df[["tenure", "monthly_charges"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["test_size"],
                                            random_state=params["random_state"])
    return X_train, X_test, y_train, y_test


def train(df, model_params, data_params):
    os.makedirs("model", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    logger.info(f"data split started")
    X_train, X_test, y_train, y_test = split_data(df, data_params)
    logger.info(f"completed data split with shape of training data = {X_train.shape}")
    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=model_params["random_state"]
    )
    logger.info("Started model fitting .................")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4)
    }

    print(f"Accuracy : {metrics['accuracy']}")
    print(f"F1 Score : {metrics['f1_score']}")
    logger.info("End of model fit and metrics calculation-----------")
    with open("model/model.pkl", "wb") as f:
            pickle.dump(model, f)

    with open("metrics/metrics.json", "w") as m:
         json.dump(metrics, m)
    return model, metrics


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--config", type=str, required=True)
    args = get_args()
    params =  load_config(args.config)
    model_params,data_params = params['model'], params['data']
    # data = generate_data(n=data_params["n_samples"],random_state=data_params["random_state"])
    data = load_data(data_params)
    logger.info(f"Generated data of shape {data.shape}")
    model, metrics = train(data, model_params=model_params, data_params=data_params)