import pandas as pd
import numpy as np
import statsmodels.api as sm

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
X_full = df[['X1', 'X2', 'X3_A', 'X3_B']]
X_full = sm.add_constant(X_full)
model_full = sm.OLS(y, X_full).fit()

# Alt modeller
# 1. Sadece X1 ve X2 içeren model
X_model1 = df[['X1', 'X2']]
X_model1 = sm.add_constant(X_model1)
model1 = sm.OLS(y, X_model1).fit()

# 2. Sadece X1 ve X3 (kategorik) içeren model
X_model2 = df[['X1', 'X3_A', 'X3_B']]
X_model2 = sm.add_constant(X_model2)
model2 = sm.OLS(y, X_model2).fit()

# 3. Sadece X2 ve X3 (kategorik) içeren model
X_model3 = df[['X2', 'X3_A', 'X3_B']]
X_model3 = sm.add_constant(X_model3)
model3 = sm.OLS(y, X_model3).fit()

# Özet tablosu
print("Model Karşılaştırması:")
print("---------------------------------------------------")
print(f"{'Model':<20} {'R²':<10} {'Düzeltilmiş R²':<15} {'Değişken Sayısı':<15}")
print(f"{'Tam Model':<20} {model_full.rsquared:<10.4f} {model_full.rsquared_adj:<15.4f} {len(model_full.params)-1:<15}")
print(f"{'Model 1 (X1, X2)':<20} {model1.rsquared:<10.4f} {model1.rsquared_adj:<15.4f} {len(model1.params)-1:<15}")
print(f"{'Model 2 (X1, X3)':<20} {model2.rsquared:<10.4f} {model2.rsquared_adj:<15.4f} {len(model2.params)-1:<15}")
print(f"{'Model 3 (X2, X3)':<20} {model3.rsquared:<10.4f} {model3.rsquared_adj:<15.4f} {len(model3.params)-1:<15}")

# R² ve düzeltilmiş R² formüllerini manuel olarak doğrulama (tam model için)
y_mean = y.mean()
SST = sum((y - y_mean)**2)  # Toplam kareler toplamı
SSR = sum((model_full.fittedvalues - y_mean)**2)  # Regresyon kareler toplamı
SSE = sum(model_full.resid**2)  # Hata kareler toplamı

R_squared = SSR / SST
# Alternatif olarak: R_squared = 1 - SSE / SST

n = len(y)  # Gözlem sayısı
k = len(model_full.params) - 1  # Bağımsız değişken sayısı (sabit terim hariç)
adj_R_squared = 1 - ((1 - R_squared) * (n - 1) / (n - k - 1))

print("\nManuel Hesaplama (Tam Model):")
print(f"Toplam Kareler Toplamı (SST): {SST:.4f}")
print(f"Regresyon Kareler Toplamı (SSR): {SSR:.4f}")
print(f"Hata Kareler Toplamı (SSE): {SSE:.4f}")
print(f"R² = SSR/SST = {R_squared:.4f}")
print(f"Düzeltilmiş R² = 1 - ((1 - R²) * (n - 1) / (n - k - 1)) = {adj_R_squared:.4f}")

# Formül doğrulaması
print("\nStatsmodels ile Karşılaştırma:")
print(f"Statsmodels R²: {model_full.rsquared:.4f}, Manuel R²: {R_squared:.4f}")
print(f"Statsmodels Düz. R²: {model_full.rsquared_adj:.4f}, Manuel Düz. R²: {adj_R_squared:.4f}")