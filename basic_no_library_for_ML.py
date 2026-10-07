import numpy as np

rng = np.random.default_rng()
noise = rng.uniform(-10, 10, size = 100)

x = np.linspace(0, 100, 100)
y = 4 + 3 * x + noise

w = 0
b = 0
n = len(x)
lr = 0.0001

for i in range(100000):
    y_pred = w * x + b
    error = np.mean((y_pred - y) ** 2)
    if (i+1) % 10000 == 0:
        print(f"error -- {error}")
        print (f"w- {w}")
        print (f"b- {b}")
    dw = (2 / n) * np.sum (x * (y_pred - y))
    # another AI asked me why w gets updated and gets to the right value so earilier than b. The answer is dw has an extra x multiplied making the update so fast.
    db = (2 / n) * np.sum (y_pred - y)

    w = w - lr * dw
    b = b - lr * db