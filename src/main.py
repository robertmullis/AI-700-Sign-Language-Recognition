import os
import glob
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FP  = os.path.join(BASE_DIR, "data", "sign_mnist_train", "sign_mnist_train.csv")
TEST_FP   = os.path.join(BASE_DIR, "data", "sign_mnist_test", "sign_mnist_test.csv")
 
def preprocess_data(filepath):
 
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
 
    df = pd.read_csv(filepath)

    df_scaled = df.copy().drop(columns=["label"]) / 255.0
    df_reshaped = df_scaled.values.reshape(-1, 28, 28, 1)

    return df_reshaped, df["label"].values

def build_model():
    pass

def train_model():
    pass

def test_model():
    pass

if __name__ == "__main__":
    X_train, y_train = preprocess_data(TRAIN_FP)
    X_test, y_test = preprocess_data(TEST_FP)

    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))