import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# ========== 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ==========

# Загружаем CSV файл
df = pd.read_csv('sunspot_data_processed.csv')

# Смотрим, какие столбцы есть в файле
print("Столбцы в файле:", df.columns.tolist())

# Берём только данные за последние 100 лет (1924-2024)
# Сначала преобразуем колонку с датой в формат datetime
df['Date'] = pd.to_datetime(df['Date (month)'])

# Фильтруем: оставляем только строки с годом >= 1924
df = df[df['Date'].dt.year >= 1924]

# Удаляем строки, где нет данных о пятнах (значение -1.0 означает пропуск)
df = df[df['Monthly mean total sunspot number'] != -1.0]
df = df.dropna(subset=['Monthly mean total sunspot number'])

# Сортируем по дате (на всякий случай)
df = df.sort_values('Date')

# Извлекаем массив значений (количество пятен) и даты
sunspots = df['Monthly mean total sunspot number'].values
dates = df['Date']

print(f"Всего точек после фильтрации: {len(sunspots)}")
print(f"Период: с {dates.min()} по {dates.max()}")

# ========== 2. ГРАФИК ИСХОДНОГО СИГНАЛА ==========

plt.figure(figsize=(14, 5))
plt.plot(dates, sunspots, 'b-', linewidth=0.7, alpha=0.7)
plt.xlabel('Год')
plt.ylabel('Число солнечных пятен')
plt.title('Солнечные пятна (последние 100 лет, 1924-2024)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sunspot_original_last100.png', dpi=150)
plt.show()

# ========== 3. FFT И СПЕКТР ==========

N = len(sunspots)  # количество точек
T_months = 1.0  # шаг между точками = 1 месяц

# Вычисляем FFT (быстрое преобразование Фурье)
yf = fft(sunspots)  # комплексные коэффициенты

# Частоты для положительных частот (от 0 до N/2)
# fftfreq возвращает частоты в единицах 1/T_months = 1/месяц
xf = fftfreq(N, T_months)[:N // 2]

# Амплитуды (модуль комплексного числа, умноженный на нормировку)
# Делим на N, потому что fft возвращает сумму, а не среднее
amplitude = 2.0 / N * np.abs(yf[:N // 2])

# Перевод частоты (1/месяц) в период (годы)
# Период в годах = 1 / (частота в 1/год) = 1 / (частота в 1/месяц * 12) = 1 / (12 * xf)
# Избегаем деления на ноль (xf[0] = 0)
with np.errstate(divide='ignore', invalid='ignore'):
    period_years = np.where(xf > 0, 1 / (12 * xf), np.inf)

# ========== ГРАФИК СПЕКТРА ==========
plt.figure(figsize=(12, 12))
mask = (period_years <= 30) & (period_years > 0)
plt.plot(period_years[mask], amplitude[mask], 'b-', linewidth=1)
plt.xlabel('Период (годы)')
plt.ylabel('Амплитуда')
plt.title('Спектр (линейный масштаб)')
plt.grid(True, alpha=0.3)
plt.xlim(0, 30)
plt.savefig('sunspot_spectrum.png', dpi=150)
plt.show()

# ========== ОПРЕДЕЛЕНИЕ ОСНОВНОГО ПЕРИОДА ==========
# Ищем максимум в диапазоне 8-15 лет (вокруг 11-летнего цикла)
mask_11 = (period_years >= 8) & (period_years <= 15)
if np.any(mask_11):
    idx_max = np.argmax(amplitude[mask_11])
    # нужно восстановить глобальный индекс
    indices = np.where(mask_11)[0]
    best_idx = indices[idx_max]
    best_period = period_years[best_idx]
    best_amp = amplitude[best_idx]
    print(f"\nОсновной период: {best_period:.2f} лет, амплитуда: {best_amp:.1f}")
else:
    # если нет данных, берём глобальный максимум
    idx_max = np.argmax(amplitude)
    best_period = period_years[idx_max]
    best_amp = amplitude[idx_max]
    print(f"\nОсновной период (глобальный): {best_period:.2f} лет, амплитуда: {best_amp:.1f}")

# ========== ДРУГИЕ ЗНАЧИМЫЕ ПЕРИОДЫ ==========
peaks, _ = find_peaks(amplitude, height=best_amp * 0.2)
print("\nДругие значимые периоды (амплитуда > 20% от главного):")
for idx in peaks:
    period = period_years[idx]
    amp = amplitude[idx]
    if 2 < period < 30 and period != best_period:
        print(f"   {period:.1f} лет, амплитуда: {amp:.1f}")

# ========== 5. ВОССТАНОВЛЕНИЕ СИГНАЛА ПО ОСНОВНЫМ ГАРМОНИКАМ ==========

def reconstruct_from_top(yf, top_indices, N):
    """
    Восстанавливает сигнал по выбранным гармоникам.

    yf: комплексный массив FFT (все коэффициенты)
    top_indices: индексы гармоник, которые нужно оставить
    N: длина исходного сигнала

    Возвращает восстановленный вещественный сигнал
    """
    yf_filtered = np.zeros_like(yf, dtype=complex)
    yf_filtered[0] = yf[0]
    for idx in top_indices:
        if idx == 0:
            continue
        yf_filtered[idx] = yf[idx]
        if idx != N // 2:
            yf_filtered[N - idx] = yf[N - idx]
    return np.fft.ifft(yf_filtered).real


# Разные числа гармоник для восстановления
N_harmonics_list = [5, 10, 20, 50]

plt.figure(figsize=(14, 10))

# График 1: весь период
plt.subplot(2, 1, 1)
plt.plot(dates, sunspots, 'b-', linewidth=0.5, alpha=0.5, label='Исходный сигнал')

for nh in N_harmonics_list:
    # Берём индексы nh самых больших амплитуд (исключая нулевую частоту)
    top_idx = np.argsort(amplitude)[-nh:]
    top_idx = top_idx[top_idx > 0]
    reconstructed = reconstruct_from_top(yf, top_idx, N)
    plt.plot(dates, reconstructed, '--', linewidth=1, label=f'{nh} гармоник', alpha=0.7)

plt.xlabel('Год')
plt.ylabel('Число солнечных пятен')
plt.title('Восстановление сигнала по основным гармоникам (1924-2024)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# График 2: последние 50 лет (для наглядности)
plt.subplot(2, 1, 2)
# Берём последние 600 месяцев = 50 лет
plt.plot(dates[-600:], sunspots[-600:], 'b-', linewidth=0.7, alpha=0.5, label='Исходный сигнал')

for nh in N_harmonics_list:
    top_idx = np.argsort(amplitude)[-nh:]
    top_idx = top_idx[top_idx > 0]
    reconstructed = reconstruct_from_top(yf, top_idx, N)
    plt.plot(dates[-600:], reconstructed[-600:], '--', linewidth=1, label=f'{nh} гармоник', alpha=0.7)

plt.xlabel('Год')
plt.ylabel('Число солнечных пятен')
plt.title('Восстановление (последние 50 лет)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sunspot_reconstructed_last100.png', dpi=150)
plt.show()

# ========== 6. КАЧЕСТВО ВОССТАНОВЛЕНИЯ (MSE) ==========

errors = []
print("\nКачество восстановления:")
for nh in N_harmonics_list:
    top_idx = np.argsort(amplitude)[-nh:]
    top_idx = top_idx[top_idx > 0]
    reconstructed = reconstruct_from_top(yf, top_idx, N)
    mse = np.mean((sunspots - reconstructed) ** 2)  # среднеквадратичная ошибка
    errors.append(mse)
    print(f"{nh:3d} гармоник: MSE = {mse:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(N_harmonics_list, errors, 'o-', linewidth=2, markersize=8)
plt.xlabel('Число гармоник')
plt.ylabel('Среднеквадратичная ошибка (MSE)')
plt.title('Зависимость качества восстановления от числа гармоник')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sunspot_error_last100.png', dpi=150)
plt.show()

