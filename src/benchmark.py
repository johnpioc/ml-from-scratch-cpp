import sys
import time
import pandas as pd
import statsmodels.api as sm

from enum import Enum
from typing import List

# ================================================================================================
# CONSTANTS AND TYPES
# ================================================================================================
TRAIN_SPLIT: float = 0.8

LIN_REG_IGNORE_COLS: List[str] = ["indus", "age"]

USAGE_MSG: str = """
    [Usage]: python3 benchmark.py [model_type]
    Model Types:
    Linear Regression: -linReg
"""

BOSTON_FILEPATH: str = "data/Boston.csv"
BOSTON_N: int = 506

class ModelType(Enum):
    NONE = 1
    LINEAR_REGRESSION = 2

class Data:
    x_train: pd.DataFrame 
    x_test: pd.DataFrame 
    y_train: pd.Series 
    y_test: pd.Series

# ================================================================================================
# HELPERS
# ================================================================================================
def throw_usage_error() -> None: raise Exception(USAGE_MSG)

def parse_cli_arguments() -> ModelType:
    model_type: ModelType = ModelType.NONE

    for i in range(1, len(sys.argv)):
        current: str = sys.argv[i]

        if model_type != ModelType.NONE: throw_usage_error()
        elif current == "-linReg": model_type = ModelType.LINEAR_REGRESSION
        else: throw_usage_error()

    if model_type == ModelType.NONE: throw_usage_error()

    return model_type

def get_boston_data(ignore: List[str]) -> Data:
    df: pd.DataFrame = pd.read_csv(BOSTON_FILEPATH, sep=",")
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop(ignore, axis=1, inplace=True)

    data: Data = Data()
    x: pd.DataFrame = df.drop("medv", axis=1)
    data.x_train = x.iloc[0: int(BOSTON_N * TRAIN_SPLIT), :].reset_index(drop=True)
    data.x_test = x.iloc[int(BOSTON_N * TRAIN_SPLIT):, :].reset_index(drop=True)


    y: pd.Series = df["medv"]
    data.y_train = y.iloc[0 : int(BOSTON_N * TRAIN_SPLIT)].reset_index(drop=True)
    data.y_test = y.iloc[int(BOSTON_N * TRAIN_SPLIT) : ].reset_index(drop=True)

    return data

def get_data(model_type: ModelType) -> Data:
    if model_type == ModelType.LINEAR_REGRESSION: return get_boston_data(LIN_REG_IGNORE_COLS)

def get_r_squared(y_pred: pd.Series, y_true: pd.Series) -> float:
    n: int = y_pred.size

    # Calculate Mean Response
    sum: float = 0.0

    for i in range(n):
        sum += y_true[i]

    y_hat: float = sum / n

    # Calculate RSS and TSS
    tss: float = 0.0
    rss: float = 0.0

    for i in range(n):
        tss += (y_true[i] - y_hat) ** 2
        rss += (y_true[i] - y_pred[i]) ** 2

    return 1.0 - (rss / tss)

def run_model(model_type: ModelType, data: Data) -> None:
    start = time.perf_counter()

    if model_type == ModelType.LINEAR_REGRESSION:
        model = sm.OLS(data.y_train, data.x_train).fit()
        end = time.perf_counter()
        runtime = end-start

        y_pred = model.predict(data.x_test)
        r_squared: float = get_r_squared(y_pred, data.y_test)

        print(f"Python statsmodels runtime: {(runtime * 1000):.4f} Milliseconds")
        print(f"Python statsmodels R Squared: {(r_squared):.2f}")

# ================================================================================================
# MAIN FUNCTION
# ================================================================================================
def main() -> None:
    model_type: ModelType = parse_cli_arguments()
    data: Data = get_data(model_type)
    run_model(model_type, data)

    return;

main()
