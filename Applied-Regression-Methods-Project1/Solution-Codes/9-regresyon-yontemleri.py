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

# Beta katsayıları ve standart hataları
beta_1 = model.params.iloc[1]  # X1 değişkeninin katsayısı
beta_2 = model.params.iloc[2]  # X2 değişkeninin katsayısı
se_beta_1 = model.bse.iloc[1]  # X1 katsayısının standart hatası
se_beta_2 = model.bse.iloc[2]  # X2 katsayısının standart hatası

# Kovaryans matrisinden beta_1 ve beta_2 arasındaki kovaryans
cov_matrix = model.cov_params()
cov_beta1_beta2 = cov_matrix.iloc[1, 2]

# H₀: β₂ = β₁ için t-testi
diff = beta_2 - beta_1  # Katsayılar arası fark
# Farkın standart hatası: SE(β₂ - β₁) = sqrt(Var(β₂) + Var(β₁) - 2*Cov(β₂,β₁))
se_diff = np.sqrt(se_beta_2**2 + se_beta_1**2 - 2*cov_beta1_beta2)
t_value = diff / se_diff  # t istatistiği

# Serbestlik derecesi ve p-değeri
df_residual = model.df_resid  # Hata serbestlik derecesi (n - k - 1)
p_value = 2 * (1 - stats.t.cdf(abs(t_value), df_residual))  # İki taraflı test

# α = 0.05 için kritik t değeri
alpha = 0.05
t_critical = stats.t.ppf(1 - alpha/2, df_residual)  # İki taraflı test için kritik değer

print("Hipotez Testi: H₀: β₂ = β₁ vs H₁: β₂ ≠ β₁")
print("---------------------------------------------------")
print(f"β₁ katsayısı (X1): {beta_1:.4f}")
print(f"β₂ katsayısı (X2): {beta_2:.4f}")
print(f"β₂ - β₁ farkı: {diff:.4f}")
print(f"Kovaryans(β₁, β₂): {cov_beta1_beta2:.6f}")
print(f"Farkın standart hatası: {se_diff:.4f}")
print(f"t istatistiği: {t_value:.4f}")
print(f"Serbestlik derecesi: {df_residual}")
print(f"p-değeri: {p_value:.4f}")
print(f"α = 0.05")
print(f"Kritik t değeri (±): {t_critical:.4f}")

if p_value < alpha:
    print("Karar: H₀ hipotezi reddedilir. β₂ ≠ β₁")
else:
    print("Karar: H₀ hipotezi reddedilemez. β₂ = β₁ olabilir")

# Güven aralığı hesaplama
confidence_level = 0.95
margin_of_error = t_critical * se_diff
lower_bound = diff - margin_of_error
upper_bound = diff + margin_of_error

print(f"\n{confidence_level * 100}% Güven Aralığı (β₂ - β₁): [{lower_bound:.4f}, {upper_bound:.4f}]")

# Güven aralığı sıfırı içeriyor mu?
if lower_bound <= 0 <= upper_bound:
    print(f"Güven aralığı 0 değerini içeriyor, bu H₀ hipotezinin reddedilemeyeceğini destekler.")
else:
    print(f"Güven aralığı 0 değerini içermiyor, bu H₀ hipotezinin reddedilmesi gerektiğini destekler.")