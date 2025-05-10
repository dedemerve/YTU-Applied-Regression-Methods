import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Veri setini oluşturalım
data = {
    'Gozlem': range(1, 13),
    'Y': [7, 8, 1, 0, 5, 3, 2, -4, 10, -3, 5, 6],
    'X1': [2, 4, 9, 11, 3, 5, 5, 10, 4, 7, 6, 1],
    'X2': [9, 2, -8, -10, 6, -6, 0, -12, 4, -2, -4, 8],
    'X3': ['A', 'A', 'C', 'C', 'B', 'B', 'C', 'C', 'A', 'C', 'B', 'B']
}

df = pd.DataFrame(data)

# Formula API kullanarak kategorik değişkeni otomatik olarak işleyelim
formula = 'Y ~ X1 + X2 + C(X3, Treatment(reference="C"))'
model = smf.ols(formula=formula, data=df).fit()

# ANOVA tablosunu oluşturalım
print("Model Özeti:")
print(model.summary())

# ANOVA tablosunu manuel olarak hesaplayalım
n = len(df)  # Gözlem sayısı
k = len(model.params) - 1  # Parametre sayısı (sabit terim hariç)

# Toplam kareler toplamı (SST)
y_mean = np.mean(df['Y'])
SST = np.sum((df['Y'] - y_mean) ** 2)

# Regresyon (Açıklanan) kareler toplamı (SSR)
y_pred = model.predict(df)
SSR = np.sum((y_pred - y_mean) ** 2)

# Hata (Artık) kareler toplamı (SSE)
SSE = np.sum((df['Y'] - y_pred) ** 2)

# Serbestlik dereceleri
df_regression = k  # Bağımsız değişken sayısı
df_residual = n - k - 1  # Toplam - bağımsız değişken sayısı - 1
df_total = n - 1  # Toplam - 1

# Ortalama kareler
MSR = SSR / df_regression
MSE = SSE / df_residual

# F istatistiği
F_statistic = MSR / MSE

# p-değeri
p_value = 1 - stats.f.cdf(F_statistic, df_regression, df_residual)

# ANOVA tablosu
anova_table = pd.DataFrame({
    'Kaynak': ['Regresyon (Model)', 'Artık (Hata)', 'Toplam'],
    'Kareler Toplamı': [SSR, SSE, SST],
    'Serbestlik Derecesi': [df_regression, df_residual, df_total],
    'Ortalama Kareler': [MSR, MSE, ''],
    'F İstatistiği': [F_statistic, '', ''],
    'p-değeri': [p_value, '', '']
})

print("\nANOVA Tablosu:")
print(anova_table)

print(f"\nF İstatistiği: {F_statistic:.4f}")
print(f"p-değeri: {p_value:.4f}")
print(f"R-kare: {model.rsquared:.4f}")
print(f"Düzeltilmiş R-kare: {model.rsquared_adj:.4f}")

# Modelin anlamlılık değerlendirmesi
alpha = 0.05  # Genellikle kullanılan anlamlılık düzeyi
if p_value < alpha:
    print(f"\nSonuç: p-değeri ({p_value:.4f}) < {alpha} olduğundan, model istatistiksel olarak anlamlıdır.")
else:
    print(f"\nSonuç: p-değeri ({p_value:.4f}) > {alpha} olduğundan, model istatistiksel olarak anlamlı değildir.")