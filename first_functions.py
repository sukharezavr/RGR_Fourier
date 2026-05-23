import numpy as np
import matplotlib.pyplot as plt

# Настройка графиков
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 12

# Количество точек для построения графиков
N_points = 1000
x = np.linspace(-np.pi, np.pi, N_points)

# Функция для вычисления частичной суммы ряда Фурье для f(x)=x
def fourier_sum_x(x, N):
    """Частичная сумма ряда Фурье для f(x)=x"""
    S = np.zeros_like(x)
    for n in range(1, N + 1):
        S += 2 * ((-1) ** (n + 1) / n) * np.sin(n * x)
    return S

# Функция для вычисления частичной суммы ряда Фурье для f(x)=|x|
def fourier_sum_abs(x, N):
    """Частичная сумма ряда Фурье для f(x)=|x|"""
    S = np.pi / 2 * np.ones_like(x)
    for k in range(1, (N // 2) + 1):
        n = 2 * k - 1  # только нечётные n
        S += (-4 / (np.pi * n ** 2)) * np.cos(n * x)
    return S

# Функция для вычисления частичной суммы ряда Фурье для sign(x)
def fourier_sum_sgn(x, N):
    """Частичная сумма ряда Фурье для sign(x)"""
    S = np.zeros_like(x)
    for k in range(1, (N // 2) + 1):
        n = 2 * k - 1  # только нечётные n
        S += (4 / (np.pi * n)) * np.sin(n * x)
    return S

# Исходные функции
def f_x(x):
    return x

def f_abs(x):
    return np.abs(x)

def f_sgn(x):
    return np.sign(x)

# Список частичных сумм для отображения
N_values = [1, 3, 5, 10, 30]

# Построение графика для f(x)=x
plt.figure(figsize=(10, 6))
plt.plot(x, f_x(x), 'k-', linewidth=2.5, label='$f(x)=x$')
for N in N_values:
    plt.plot(x, fourier_sum_x(x, N), '--', label=f'$S_{N}(x)$', alpha=0.7)
plt.xlabel('$x$')
plt.ylabel('$f(x), S_N(x)$')
plt.title('Ряд Фурье для $f(x)=x$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-np.pi, np.pi)
plt.tight_layout()
plt.savefig('fourier_x.png', dpi=150)
plt.show()

# Построение графика для f(x)=|x|
plt.figure(figsize=(10, 6))
plt.plot(x, f_abs(x), 'k-', linewidth=2.5, label='$f(x)=|x|$')
for N in N_values:
    plt.plot(x, fourier_sum_abs(x, N), '--', label=f'$S_{N}(x)$', alpha=0.7)
plt.xlabel('$x$')
plt.ylabel('$f(x), S_N(x)$')
plt.title('Ряд Фурье для $f(x)=|x|$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-np.pi, np.pi)
plt.tight_layout()
plt.savefig('fourier_absx.png', dpi=150)
plt.show()

# Построение графика для sign(x)
plt.figure(figsize=(10, 6))
plt.plot(x, f_sgn(x), 'k-', linewidth=2.5, label='$\\operatorname{sgn}(x)$')
for N in N_values:
    plt.plot(x, fourier_sum_sgn(x, N), '--', label=f'$S_{N}(x)$', alpha=0.7)
plt.xlabel('$x$')
plt.ylabel('$f(x), S_N(x)$')
plt.title('Ряд Фурье для $\\operatorname{sgn}(x)$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-np.pi, np.pi)
plt.ylim(-1.5, 1.5)
plt.tight_layout()
plt.savefig('fourier_sgn.png', dpi=150)
plt.show()