import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

temp = np.linspace(0, 100, 1000)
params = {
    'Triangular': (fuzz.trimf, [[0, 0, 30], [20, 50, 80], [60, 100, 100]]),
    'Trapezoidal': (fuzz.trapmf, [[0, 0, 20, 40], [30, 50, 70, 90], [80, 100, 100, 100]]),
    'Gaussian': (fuzz.gaussmf, [[15, 10], [50, 15], [85, 10]]),
    'Sigmoid': (fuzz.sigmf, [[20, -0.1], [50, 0.1], [80, 0.1]])
}
colors = ['b', 'g', 'r']
labels = ['Cold', 'Warm', 'Hot']

plt.figure(figsize=(15, 10))
for i, (title, (func, values)) in enumerate(params.items()):
    plt.subplot(2, 2, i + 1)
    for val, color, label in zip(values, colors, labels):
        if title == 'Triangular' or title == 'Trapezoidal':
            y = func(temp, val)
        elif title == 'Sigmoid' and label == 'Warm':
            y = fuzz.sigmf(temp, 50, 0.1) - fuzz.sigmf(temp, 50, -0.1)
        else:
            y = func(temp, *val)
        plt.plot(temp, y, color, label=f'{label} ({title})')
    plt.title(f'{title} Membership Functions')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Membership Degree')
    plt.legend()

plt.tight_layout()
plt.show()
