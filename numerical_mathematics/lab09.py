import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
def integral(a, b):
    return (2*b*np.log(b) - 2*b) - (2*a*np.log(a) - 2*a)


def middle_point(f, a, b, n):
    h = (b - a) / n
    return h * np.sum([f(a + i*h + h/2) for i in range(n-2)])


def trapez():
    pass


def simpson():
    pass

sp.init_printing()

xs = np.linspace(0, 3)
fig, ax = plt.subplots(figsize=(16, 16))
ax.grid(True)

# thicker 0-axis
ax.axhline(0, color='black', linewidth=2)
ax.axvline(0, color='black', linewidth=2)

# vertical dashed lines from x-axis to the y-values
ax.plot([1, 1], [0, np.log(1)], linestyle='--', color='gray')
ax.plot([2, 2], [0, np.log(4)], linestyle='--', color='gray')

# circles around intersection points
ax.scatter(2, 0, color='red', edgecolor='black')
ax.scatter(2, np.log(4), color='red', edgecolor='black')

ax.plot(xs, np.log(xs**2))
print('real value', integral(1, 2))
for n in range(1, 16):
    print(f'n = {n}', middle_point(lambda x: np.log(x**2), 1, 2, n))
plt.show()


x = sp.symbols('x')
f = sp.ln(x**2)
err = 1e-5




