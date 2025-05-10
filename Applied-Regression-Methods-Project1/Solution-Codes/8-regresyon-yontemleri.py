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

# H₀: β₂ = 1 için t-testi
null_value = 1  # Sıfır hipotezinde belirtilen değer
t_value = (beta_2 - null_value) / se_beta_2  # t istatistiği

# Serbestlik derecesi ve p-değeri
df_residual = model.df_resid  # Hata serbestlik derecesi (n - k - 1)
p_value = 2 * (1 - stats.t.cdf(abs(t_value), df_residual))  # İki taraflı test

# α = 0.05 için kritik t değeri
alpha = 0.05
t_critical = stats.t.ppf(1 - alpha/2, df_residual)  # İki taraflı test için kritik değer

print("Hipotez Testi: H₀: β₂ = 1 vs H₁: β₂ ≠ 1")
print("---------------------------------------------------")
print(f"β₂ katsayısı: {beta_2:.4f}")
print(f"Standart hata: {se_beta_2:.4f}")
print(f"Sıfır hipotezi değeri: {null_value}")
print(f"t istatistiği: {t_value:.4f}")
print(f"Serbestlik derecesi: {df_residual}")
print(f"p-değeri: {p_value:.4f}")
print(f"α = 0.05")
print(f"Kritik t değeri (±): {t_critical:.4f}")

if p_value < alpha:
    print("Karar: H₀ hipotezi reddedilir. β₂ ≠ 1")
else:
    print("Karar: H₀ hipotezi reddedilemez. β₂ = 1 olabilir")

# Güven aralığı hesaplama
confidence_level = 0.95
margin_of_error = t_critical * se_beta_2
lower_bound = beta_2 - margin_of_error
upper_bound = beta_2 + margin_of_error

print(f"\n{confidence_level * 100}% Güven Aralığı: [{lower_bound:.4f}, {upper_bound:.4f}]")

# Güven aralığı sıfır hipotezinde belirtilen değeri içeriyor mu?
if lower_bound <= null_value <= upper_bound:
    print(f"Güven aralığı {null_value} değerini içeriyor, bu H₀ hipotezinin reddedilemeyeceğini destekler.")
else:
    print(f"Güven aralığı {null_value} değerini içermiyor, bu H₀ hipotezinin reddedilmesi gerektiğini destekler.")