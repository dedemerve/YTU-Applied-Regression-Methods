import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Veri setini oluşturalım
data = {
    'Gozlem': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'X1': [2, 4, 9, 11, 3, 5, 5, 10, 4, 7, 6, 1],
    'X2': [9, 2, -8, -10, 6, -6, 0, -12, 4, -2, -4, 8],
    'İlaç': ['A', 'A', 'C', 'C', 'B', 'B', 'C', 'C', 'A', 'C', 'B', 'B'],
    'Y': [7, 8, 1, 0, 5, 3, 2, -4, 10, -3, 5, 6]
}

df = pd.DataFrame(data)

# Dummy değişkenler oluştur (İlaç türü için)
# A ilaç türünü referans kategori olarak alıyoruz
df['D1'] = (df['İlaç'] == 'B').astype(int)  # B için dummy
df['D2'] = (df['İlaç'] == 'C').astype(int)  # C için dummy

# OLS için modeli kur
X = df[['X1', 'X2', 'D1', 'D2']]
X = sm.add_constant(X)  # Sabit terim (β₀) ekle
y = df['Y']

# OLS ile tahmin
model_ols = sm.OLS(y, X).fit()
print("OLS Sonuçları:")
print(model_ols.summary())

# Kalıntıları hesapla
residuals = model_ols.resid
n = len(residuals)  # Gözlem sayısı
k = X.shape[1]      # Parametre sayısı (sabit terim dahil)
df_residual = n - k  # Serbestlik derecesi

# Hata terimlerinin varyans tahmini
residual_variance = np.sum(residuals**2) / df_residual
print(f"\nKalıntıların varyans tahmini (s²): {residual_variance:.4f}")

# H₀: σ² = 4 hipotez testi
# Chi-kare test istatistiği: (n-k)s²/σ₀² ~ χ²(n-k)
null_variance = 4  # H₀ altında varsayılan varyans değeri
chi_square_stat = (df_residual * residual_variance) / null_variance
p_value_right = 1 - stats.chi2.cdf(chi_square_stat, df_residual)  # sağ kuyruk testi için
p_value_left = stats.chi2.cdf(chi_square_stat, df_residual)       # sol kuyruk testi için

# İki yönlü test için p-değeri
p_value_two_sided = 2 * min(p_value_right, p_value_left)

print(f"\nHipotez Testi: H₀: σ² = 4 vs H₁: σ² ≠ 4")
print(f"Serbestlik derecesi: {df_residual}")
print(f"Chi-kare test istatistiği: {chi_square_stat:.4f}")
print(f"Sol kuyruk p-değeri: {p_value_left:.4f}")
print(f"Sağ kuyruk p-değeri: {p_value_right:.4f}")
print(f"İki yönlü test p-değeri: {p_value_two_sided:.4f}")

# Kritik değerler (α = 0.05)
alpha = 0.05
critical_value_lower = stats.chi2.ppf(alpha/2, df_residual)
critical_value_upper = stats.chi2.ppf(1-alpha/2, df_residual)

print(f"\nα = 0.05 için kritik değerler:")
print(f"Alt kritik değer: {critical_value_lower:.4f}")
print(f"Üst kritik değer: {critical_value_upper:.4f}")

# Karar
print("\nKarar:")
if chi_square_stat < critical_value_lower or chi_square_stat > critical_value_upper:
    print(f"H₀ reddedilir (p-değeri = {p_value_two_sided:.4f} < 0.05)")
    print("Varyansın 4'e eşit olduğu hipotezi reddedilir.")
else:
    print(f"H₀ reddedilemez (p-değeri = {p_value_two_sided:.4f} ≥ 0.05)")
    print("Varyansın 4'e eşit olduğu hipotezi reddedilemez.")