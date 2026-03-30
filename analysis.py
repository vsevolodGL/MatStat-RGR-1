import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===== загрузка данных =====
df = pd.read_csv("RGR1_D-1_X1-X4.csv")

X1 = df["X1"].values
X2 = df["X2"].values
X3 = df["X3"].values
X4 = df["X4"].values

os.makedirs("plots", exist_ok=True)

# ===== правило Скотта =====
def scott_bin_width(x):
    n = len(x)
    std = np.std(x, ddof=1)
    h = 3.5 * std * (n ** (-1/3))
    return h

def scott_bins(x):
    h = scott_bin_width(x)
    k = int(np.ceil((np.max(x) - np.min(x)) / h))
    return k, h

# ===== ECDF =====
def ecdf(x):
    x_sorted = np.sort(x)
    y = np.arange(1, len(x)+1) / len(x)
    return x_sorted, y

# ===== анализ =====
def analyze(x, name):
    print(f"\n===== {name} =====")

    mean = np.mean(x)
    var1 = np.var(x)
    var2 = np.var(x, ddof=1)
    disp = np.std(x)
    median = np.median(x)
    q1, q3 = np.quantile(x, [0.25, 0.75]).tolist()

    print(f"Среднее = {mean}")
    print(f"Дисперсия смещённая = {var1}")
    print(f"Дисперсия несмещённая = {var2}")
    print(f"Стандартное отклонение смещённая = {disp}")
    print(f"Медиана = {median}")
    print(f"1-й и 3-й квартили = {q1, q3}")
    print(f"Миниальное значение = {np.min(x)}")

    # правило Скотта
    k, h = scott_bins(x)
    print(f"Scott h = {h}")
    print(f"bins k = {k}")

    # гистограмма
    plt.figure()
    plt.hist(x, bins=k, edgecolor='black')
    plt.title(f"{name} histogram (Scott)")
    plt.savefig(f"plots/{name}_hist.png")
    plt.close()

    # ECDF
    x_ecdf, y_ecdf = ecdf(x)
    plt.figure()
    plt.step(x_ecdf, y_ecdf, where="post")
    plt.title(f"{name} ECDF")
    plt.savefig(f"plots/{name}_ecdf.png")
    plt.close()

# ===== запуск =====
for i, X in enumerate([X1, X2, X3, X4], start=1):
    analyze(X, f"x{i}")