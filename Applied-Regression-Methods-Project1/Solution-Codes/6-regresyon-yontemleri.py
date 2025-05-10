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

# 1. TAM MODEL (X1, X2, X3_A, X3_B)
X_full = df[['X1', 'X2', 'X3_A', 'X3_B']]
X_full = sm.add_constant(X_full)
model_full = sm.OLS(y, X_full).fit()

# 2. KISITLI MODEL (X2 parametresi çıkarılmış)
X_restricted = df[['X1', 'X3_A', 'X3_B']]
X_restricted = sm.add_constant(X_restricted)
model_restricted = sm.OLS(y, X_restricted).fit()

# Tam ve kısıtlı modellerin kalıntı kareler toplamı
SSE_full = sum(model_full.resid**2)
SSE_restricted = sum(model_restricted.resid**2)

# Serbestlik dereceleri
df_full = len(y) - len(model_full.params)  # n - k_full
df_restricted = len(y) - len(model_restricted.params)  # n - k_restricted
df_numerator = df_restricted - df_full  # Kısıt sayısı (bu durumda 1)

# Kısmi F testi istatistiği
F_statistic = ((SSE_restricted - SSE_full) / df_numerator) / (SSE_full / df_full)

# p-değeri
p_value = 1 - stats.f.cdf(F_statistic, df_numerator, df_full)

print("Kısmi F Testi: H₀: β₂ = 0 vs H₁: β₂ ≠ 0")
print("---------------------------------------------------")
print(f"Tam Model R²: {model_full.rsquared:.4f}")
print(f"Kısıtlı Model R²: {model_restricted.rsquared:.4f}")
print(f"Tam Model SSE: {SSE_full:.4f}")
print(f"Kısıtlı Model SSE: {SSE_restricted:.4f}")
print(f"F İstatistiği: {F_statistic:.4f}")
print(f"Serbestlik Dereceleri: ({df_numerator}, {df_full})")
print(f"p-değeri: {p_value:.4f}")
print(f"α = 0.05")

if p_value < 0.05:
    print("Karar: H₀ hipotezi reddedilir. β₂ ≠ 0")
else:
    print("Karar: H₀ hipotezi reddedilemez. β₂ = 0")

# Ayrıca β₂ katsayısının t-testi ile test edilmesi (karşılaştırma için)
print("\nKarşılaştırma için β₂ katsayısının t-testi:")
print(f"β₂ katsayısı: {model_full.params.iloc[2]:.4f}")
print(f"Standart hata: {model_full.bse.iloc[2]:.4f}")
t_value = model_full.params.iloc[2] / model_full.bse.iloc[2]
p_value_t = 2 * (1 - stats.t.cdf(abs(t_value), df_full))
print(f"t-değeri: {t_value:.4f}")
print(f"p-değeri: {p_value_t:.4f}")

if p_value_t < 0.05:
    print("Karar: H₀ hipotezi reddedilir. β₂ ≠ 0")
else:
    print("Karar: H₀ hipotezi reddedilemez. β₂ = 0")

# Not: Kısmi F testi ve t-testi sonuçları aynı olmalıdır (tek parametreli kısıt için)
# F = t² durumunda
print(f"\nDoğrulama: t² = {t_value**2:.4f}, F = {F_statistic:.4f}")