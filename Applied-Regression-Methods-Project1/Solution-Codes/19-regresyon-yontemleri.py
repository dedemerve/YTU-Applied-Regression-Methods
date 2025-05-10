import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt

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

print("Veri Seti:")
print(df)

# OLS için modeli kur
X = df[['X1', 'X2', 'D1', 'D2']]
X = sm.add_constant(X)  # Sabit terim (β₀) ekle
y = df['Y']

# 1. OLS ile tahmin
model_ols = sm.OLS(y, X).fit()
print("\nOLS Sonuçları:")
print(model_ols.summary())

# OLS kalıntılarını hesapla
df['residuals'] = model_ols.resid
df['abs_residuals'] = np.abs(model_ols.resid)
df['residuals_squared'] = model_ols.resid**2

# Kalıntılar grafiği
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['residuals'])
plt.axhline(y=0, color='r', linestyle='-')
plt.title('OLS Kalıntıları')
plt.xlabel('Gözlem')
plt.ylabel('Kalıntı')
print("\nOLS kalıntıları hesaplandı ve grafiği oluşturuldu.")

# Heteroskedastisite testi (Breusch-Pagan testi)
bp_test = het_breuschpagan(model_ols.resid, model_ols.model.exog)
labels = ['LM İstatistiği', 'LM p-değeri', 'F İstatistiği', 'F p-değeri']
bp_results = dict(zip(labels, bp_test))
print("\nBreusch-Pagan Heteroskedastisite Testi:")
for key, value in bp_results.items():
    print(f"{key}: {value:.4f}")

# 2. WLS ile tahmin
# Kalıntı karelerinin tersini ağırlık olarak kullanıyoruz
weights = 1 / (df['residuals_squared'] + 0.0001)  # 0'a bölünmeyi önlemek için küçük bir değer ekliyoruz
model_wls = sm.WLS(y, X, weights=weights).fit()

print("\nWLS (Ağırlıklı En Küçük Kareler) Sonuçları:")
print(model_wls.summary())

# OLS ve WLS sonuçlarını karşılaştır
comparison = pd.DataFrame({
    'OLS Katsayıları': model_ols.params,
    'OLS Std. Hata': model_ols.bse,
    'WLS Katsayıları': model_wls.params,
    'WLS Std. Hata': model_wls.bse
})

print("\nOLS ve WLS Karşılaştırması:")
print(comparison)

# R-kare karşılaştırması
print(f"\nOLS R-kare: {model_ols.rsquared:.4f}")
print(f"WLS R-kare: {model_wls.rsquared:.4f}")

# AIC ve BIC karşılaştırması
print(f"\nOLS AIC: {model_ols.aic:.4f}")
print(f"WLS AIC: {model_wls.aic:.4f}")
print(f"OLS BIC: {model_ols.bic:.4f}")
print(f"WLS BIC: {model_wls.bic:.4f}")