import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Veri setini oluşturma
data = {
    'X1': [2, 4, 9, 11, 3, 5, 5, 10, 4, 7, 6, 1],
    'X2': [9, 2, -8, -10, 6, -6, 0, -12, 4, -2, -4, 8],
    'X3': ['A', 'A', 'C', 'C', 'B', 'B', 'C', 'C', 'A', 'C', 'B', 'B'],
    'Y': [7, 8, 1, 0, 5, 3, 2, -4, 10, -3, 5, 6]
}

# Veri tiplerini açıkça belirterek DataFrame oluşturma
df = pd.DataFrame({
    'X1': pd.Series(data['X1'], dtype=float),
    'X2': pd.Series(data['X2'], dtype=float),
    'X3': pd.Series(data['X3'], dtype='category'),
    'Y': pd.Series(data['Y'], dtype=float)
})

# Manuel olarak dummy değişkenler oluşturma (C'yi referans olarak alıyoruz)
df['X3_A'] = (df['X3'] == 'A').astype(float)
df['X3_B'] = (df['X3'] == 'B').astype(float)

# Bağımlı değişken
y = df['Y']

# Tam model (X1, X2, X3_A, X3_B)
X = df[['X1', 'X2', 'X3_A', 'X3_B']]
X = sm.add_constant(X)  # Sabit terim ekleme
model = sm.OLS(y, X).fit()

# β₂ katsayısı ve standart hatası
beta_2 = model.params.iloc[2]  # X2 değişkeninin katsayısı (2. indeks)
se_beta_2 = model.bse.iloc[2]  # X2 katsayısının standart hatası

# α = 0.05 için kritik t değeri
alpha = 0.05
df_residual = model.df_resid  # Hata serbestlik derecesi (n - k - 1)
t_critical = stats.t.ppf(1 - alpha/2, df_residual)  # İki taraflı test için kritik değer

# %95 güven aralığı hesaplama
margin_of_error = t_critical * se_beta_2
lower_bound = beta_2 - margin_of_error
upper_bound = beta_2 + margin_of_error

print("β₂ için %95 Güven Aralığı:")
print("---------------------------------------------------")
print(f"β₂ katsayısı: {beta_2:.4f}")
print(f"Standart hata: {se_beta_2:.4f}")
print(f"Kritik t değeri (±): {t_critical:.4f}")
print(f"Hata payı: {margin_of_error:.4f}")
print(f"%95 güven aralığı: [{lower_bound:.4f}, {upper_bound:.4f}]")

# Alternatif yöntem - model.conf_int() kullanarak
conf_int = model.conf_int(alpha=0.05)
print("\nStatsmodels conf_int() kullanarak:")
print(f"%95 güven aralığı (β₂): [{conf_int.iloc[2, 0]:.4f}, {conf_int.iloc[2, 1]:.4f}]")

# Grafiksel gösterim
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
x = np.linspace(beta_2 - 4*se_beta_2, beta_2 + 4*se_beta_2, 1000)
y = stats.t.pdf(x, df=df_residual, loc=beta_2, scale=se_beta_2)

plt.plot(x, y)
plt.fill_between(x, y, where=((x >= lower_bound) & (x <= upper_bound)), alpha=0.3, color='blue')
plt.axvline(x=beta_2, color='red', linestyle='-', label=f'β₂ = {beta_2:.4f}')
plt.axvline(x=lower_bound, color='green', linestyle='--', label=f'Alt Sınır = {lower_bound:.4f}')
plt.axvline(x=upper_bound, color='green', linestyle='--', label=f'Üst Sınır = {upper_bound:.4f}')
plt.axvline(x=0, color='black', linestyle=':', label='β₂ = 0')
plt.axvline(x=1, color='purple', linestyle=':', label='β₂ = 1')

plt.title('β₂ için %95 Güven Aralığı')
plt.xlabel('β₂ Değeri')
plt.ylabel('Olasılık Yoğunluğu')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()