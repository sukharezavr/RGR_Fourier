import math
import numpy as np
import matplotlib.pyplot as plt
import time

def simpson(f, n, a, b):
    """Метод Симпсона для численного интегрирования"""
    if n % 2 != 0:
        n += 1  # чтобы n было чётным
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [f(xi) for xi in x]
    return h / 3 * (y[0] + y[-1] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-1:2]))


# ========== КОЭФФИЦИЕНТЫ ФУРЬЕ ЧИСЛЕННО ==========
def fourier_coeffs_simpson(f, n_max, a, b, N_int):
    """
    Вычисляет коэффициенты Фурье численно (методом Симпсона)
    f: функция
    n_max: максимальный номер гармоники
    a, b: границы отрезка (обычно -pi, pi)
    N_int: число разбиений для интегрирования
    """
    L = b - a  # длина периода
    a0 = (2 / L) * simpson(f, N_int, a, b)

    an = []
    bn = []

    for n in range(1, n_max + 1):
        # Подынтегральные функции
        def f_cos(x):
            return f(x) * math.cos(2 * math.pi * n * x / L)

        def f_sin(x):
            return f(x) * math.sin(2 * math.pi * n * x / L)

        an_n = (2 / L) * simpson(f_cos, N_int, a, b)
        bn_n = (2 / L) * simpson(f_sin, N_int, a, b)

        an.append(an_n)
        bn.append(bn_n)

    return a0, an, bn

# ========== ПОСТРОЕНИЕ ЧАСТИЧНОЙ СУММЫ ==========
def partial_sum(x, a0, an, bn, a, b, N):
    """
    Частичная сумма ряда Фурье по численно найденным коэффициентам
    """
    L = b - a
    result = a0 / 2
    for n in range(1, N + 1):
        result += an[n - 1] * math.cos(2 * math.pi * n * x / L) + bn[n - 1] * math.sin(2 * math.pi * n * x / L)
    return result



# ========== ПРИМЕР ДЛЯ ЗНАКОВОЙ ФУНКЦИИ ==========
def f_sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

"""
# Параметры
a, b = -math.pi, math.pi
n_max = 30  # максимальная гармоника для расчёта коэффициентов
N_int = 1024  # число разбиений для численного интегрирования

# Численно вычисляем коэффициенты
print("Вычисление коэффициентов Фурье численно (Симпсон)...")
a0, an, bn = fourier_coeffs_numerical(f_sgn, n_max, a, b, N_int)
print(f"a0 = {a0:.6f}")
print(f"b1 = {bn[0]:.6f} (аналитический: {4/math.pi:.6f})")
print(f"b3 = {bn[2]:.6f} (аналитический: {4/(3*math.pi):.6f})")

# Строим графики частичных сумм
x_plot = np.linspace(a, b, 1000)
y_true = [f_sgn(x) for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_true, 'k-', linewidth=2.5, label='sgn(x)')

# Разные N для частичных сумм
N_values = [1, 3, 5, 10, 30]
for N in N_values:
    y_sum = [partial_sum(x, a0, an, bn, a, b, N) for x in x_plot]
    plt.plot(x_plot, y_sum, '--', label=f'S_{N}(x)', alpha=0.7)  # без $

plt.xlabel('x')
plt.ylabel('f(x), S_N(x)')
plt.title('Аппроксимация знаковой функции (коэффициенты найдены численно)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(a, b)
plt.ylim(-1.5, 1.5)
plt.tight_layout()
plt.savefig('fourier_numerical_sgn.png', dpi=150)
plt.show()

"""

# ========== ВЫЧИСЛЕНИЕ КОЭФФИЦИЕНТОВ ЧЕРЕЗ FFT ==========
def fourier_coeffs_fft(f, n_max, a, b, N):
    """
    Вычисляет коэффициенты Фурье через FFT
    f: функция
    n_max: максимальная гармоника
    a, b: границы отрезка
    N: число точек дискретизации (должно быть степенью 2)
    """
    T = b - a  # период

    # Дискретизация
    x = np.linspace(a, b, N, endpoint=False)
    y = np.array([f(xi) for xi in x])

    # FFT
    fft_vals = np.fft.fft(y) / N

    # a0
    a0 = 2 * fft_vals[0].real

    an = []
    bn = []
    for n in range(1, n_max + 1):
        if n < len(fft_vals):
            c_n = fft_vals[n]
            an.append(2 * c_n.real)
            bn.append(2 * c_n.imag)
        else:
            an.append(0.0)       # нет данных -> ноль
            bn.append(0.0)

    return a0, an, bn


# ========== ПАРАМЕТРЫ ==========
a, b = -math.pi, math.pi
n_max = 30
# N - степени двойки (одинаковые для обоих методов)
N_values = [64, 128, 256, 512, 1024, 2048, 4096]


# Аналитический b_n для проверки точности
def b_analytical(n):
    if n % 2 == 0:
        return 0
    return 4 / (math.pi * n)



print("=" * 100)
print("СРАВНЕНИЕ МЕТОДОВ: СИМПСОН vs FFT (N — степени двойки)")
print("=" * 100)
print(f"{'N':>8} | {'Время Симпсон (с)':>18} | {'Время FFT (с)':>15} | {'Ошибка Симпсон':>18} | {'Ошибка FFT':>15}")
print("-" * 100)

# Для хранения результатов
time_simpson_list = []
time_fft_list = []
error_simpson_list = []
error_fft_list = []

for N in N_values:
    # Симпсон
    start = time.time()
    a0_s, an_s, bn_s = fourier_coeffs_simpson(f_sgn, n_max, a, b, N)
    time_s = time.time() - start

    # FFT
    start = time.time()
    a0_f, an_f, bn_f = fourier_coeffs_fft(f_sgn, n_max, a, b, N)
    time_f = time.time() - start

    # Средняя ошибка для b_n (первые 10 нечётных гармоник)
    err_s = 0
    err_f = 0
    count = 0
    for n in range(1, min(20, n_max + 1), 2):
        b_true = b_analytical(n)
        err_s += abs(b_true - bn_s[n - 1])
        err_f += abs(b_true - bn_f[n - 1])
        count += 1
    err_s /= count
    err_f /= count

    time_simpson_list.append(time_s)
    time_fft_list.append(time_f)
    error_simpson_list.append(err_s)
    error_fft_list.append(err_f)

    print(f"{N:8d} | {time_s:18.6f} | {time_f:15.6f} | {err_s:18.2e} | {err_f:15.2e}")

# ========== ГРАФИКИ ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# График времени
ax1.plot(N_values, time_simpson_list, 'o-', label='Симпсон', linewidth=2, markersize=8)
ax1.plot(N_values, time_fft_list, 's-', label='FFT', linewidth=2, markersize=8)
ax1.set_xlabel('Число разбиений / точек N (степень двойки)')
ax1.set_ylabel('Время выполнения (сек)')
ax1.set_title('Сравнение скорости')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График ошибки
ax2.semilogy(N_values, error_simpson_list, 'o-', label='Симпсон', linewidth=2, markersize=8)
ax2.semilogy(N_values, error_fft_list, 's-', label='FFT', linewidth=2, markersize=8)
ax2.set_xlabel('Число разбиений / точек N (степень двойки)')
ax2.set_ylabel('Средняя погрешность b_n')
ax2.set_title('Сравнение точности (лог. шкала)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simpson_vs_fft_comparison.png', dpi=150)
plt.show()


# ========== ПУНКТ 3: СРАВНЕНИЕ АНАЛИТИЧЕСКОГО, СИМПСОНА И FFT ==========
print("\n" + "=" * 100)
print("ПУНКТ 3: СРАВНЕНИЕ АНАЛИТИЧЕСКОГО РАЗЛОЖЕНИЯ, СИМПСОНА И FFT")
print("=" * 100)

# Аналитическая частичная сумма для знаковой функции
def partial_sum_analytical(x, N):
    S = 0
    for n in range(1, N+1, 2):   # шаг 2 → только нечётные
        S += (4 / (math.pi * n)) * math.sin(n * x)
    return S

# Параметры для сравнения
N_comp = 50          # количество гармоник
N_disc = 2048         # количество точек для диксретизации FFT и разбиений для симпсона
N_points = 1000      # точек для гладкого графика
x_plot = np.linspace(a, b, N_points)

print(f"Сравнение при N_гармоник = {N_comp}, N_точек дискретизации = {N_disc}")

# Вычисляем коэффициенты
a0_s, an_s, bn_s = fourier_coeffs_simpson(f_sgn, N_comp, a, b, N_disc)
a0_f, an_f, bn_f = fourier_coeffs_fft(f_sgn, N_comp, a, b, N_disc)

# Частичные суммы
y_analytical = [partial_sum_analytical(x, N_comp) for x in x_plot]
y_simpson = [partial_sum(x, a0_s, an_s, bn_s, a, b, N_comp) for x in x_plot]
y_fft = [partial_sum(x, a0_f, an_f, bn_f, a, b, N_comp) for x in x_plot]
y_true = [f_sgn(x) for x in x_plot]

# График 1: общий вид
plt.figure(figsize=(12, 6))
plt.plot(x_plot, y_true, 'k-', linewidth=2, label='Исходная')
plt.plot(x_plot, y_analytical, 'b--', linewidth=1.5, label='Аналитическое', alpha=0.8)
plt.plot(x_plot, y_simpson, 'g--', linewidth=1.5, label='Симпсон', alpha=0.8)
plt.plot(x_plot, y_fft, 'r--', linewidth=1.5, label='FFT', alpha=0.8)
plt.xlabel('$x$')
plt.ylabel('$S_N(x)$')
plt.title(f'Сравнение методов (N_comp = {N_comp}, N_disc = {N_disc})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(a, b)
plt.ylim(-1.5, 1.5)
plt.tight_layout()
plt.savefig('comparison_three_methods.png', dpi=150)
plt.show()

# ========== ГРАФИК ОШИБКИ ПРИБЛИЖЕНИЯ ФУНКЦИИ ==========

# Точка, в которой оцениваем ошибку (не в разрыве)
x_test = 1.0
true_val = f_sgn(x_test)

N_harm_list = list(range(1, N_comp, 2))  # нечётные от 1 до N_comp
error_analytical = []
error_simpson = []
error_fft = []

# Используем коэффициенты, полученные при N_disc
a0_s, an_s, bn_s = fourier_coeffs_simpson(f_sgn, max(N_harm_list), a, b, N_disc)
a0_f, an_f, bn_f = fourier_coeffs_fft(f_sgn, max(N_harm_list), a, b, N_disc)

for N in N_harm_list:
    # Аналитическая частичная сумма
    S_analyt = partial_sum_analytical(x_test, N)
    error_analytical.append(abs(true_val - S_analyt))

    # Симпсон
    S_simp = partial_sum(x_test, a0_s, an_s, bn_s, a, b, N)
    error_simpson.append(abs(true_val - S_simp))

    # FFT
    S_fft = partial_sum(x_test, a0_f, an_f, bn_f, a, b, N)
    error_fft.append(abs(true_val - S_fft))

# График ошибки
plt.figure(figsize=(10, 6))
plt.loglog(N_harm_list, error_analytical, 'o-', label='Аналитическое', linewidth=2, markersize=6)
plt.loglog(N_harm_list, error_simpson, 's-', label=f'Симпсон (N_disc={N_disc})', linewidth=2, markersize=6)
plt.loglog(N_harm_list, error_fft, '^-', label=f'FFT (N_disc={N_disc})', linewidth=2, markersize=6)
plt.xlabel('Число гармоник N')
plt.ylabel('Ошибка в точке x=1')
plt.title('Скорость сходимости ряда Фурье для $\\operatorname{sgn}(x)$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('function_error_vs_harmonics.png', dpi=150)
plt.show()

# ========== ТАБЛИЦА СРАВНЕНИЯ ОШИБОК ==========
print("\n" + "=" * 80)
print("ТАБЛИЦА СРАВНЕНИЯ ОШИБОК (аналитический vs Симпсон vs FFT)")
print("=" * 80)
print(f"{'N гармоник':>12} | {'Аналит. ошибка':>18} | {'Симпсон ошибка':>18} | {'FFT ошибка':>18}")
print("-" * 75)

for i, N in enumerate(N_harm_list):
    print(f"{N:12d} | {error_analytical[i]:18.2e} | {error_simpson[i]:18.2e} | {error_fft[i]:18.2e}")
