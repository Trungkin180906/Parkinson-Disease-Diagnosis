import os
import numpy as np
import pandas as pd

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../dataset/draw/Spiral_HandPD.csv"
)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    df = pd.read_csv(CSV_PATH, header=None, skiprows=1)

    # Cột 3 -> 11: 9 đặc trưng
    X = df.iloc[:, 3:12].values.astype(float)

    # Cột 2: nhãn
    y = df.iloc[:, 2].values.astype(int)

    # Đổi nhãn 1/2 thành 0/1
    y = np.where(y == 1, 0, 1)

    np.save(os.path.join(SAVE_DIR, "X_drawing.npy"), X)
    np.save(os.path.join(SAVE_DIR, "y_drawing.npy"), y)

    print("Đã xử lý dữ liệu Drawing")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Labels:", np.unique(y, return_counts=True))


if __name__ == "__main__":
    main()