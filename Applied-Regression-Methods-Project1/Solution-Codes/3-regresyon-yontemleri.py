import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Veri setini oluşturma
data = {
    'X1': [2, 4, 9, 11, 3, 5, 5, 10, 4, 7, 6, 1],
    'X2': [9, 2, -8, -10, 6, -6, 0, -12, 4, -2, -4, 8],
    'X3': ['A', 'A', 'C', 'C', 'B', 'B', 'C', 'C', 'A', 'C', 'B', 'B'],
    'Y': [7, 8, 1, 0, 5, 3, 2, -4, 10, -3, 5, 6]
}
df = pd.DataFrame(data)

# Sadece X3'ün Y üzerindeki etkisini test eden model
model_x3 = ols('Y ~ C(X3)', data=df).fit()

# X3 için ANOVA tablosu
anova_table_x3 = anova_lm(model_x3)
print("X3 Değişkeni için ANOVA Tablosu:")
print(anova_table_x3)

# Hipotez testi (F-testi) - X3 için
f_value = anova_table_x3.loc['C(X3)', 'F']  # ANOVA tablosundaki X3 F değeri
p_value = anova_table_x3.loc['C(X3)', 'PR(>F)']  # ANOVA tablosundaki X3 p değeri
alpha = 0.05

print("\nX3 Değişkeni İçin Anlamlılık Testi:")
print(f"H₀: X3 değişkeninin farklı düzeyleri (A, B, C) arasında Y açısından fark yoktur.")
print(f"H₁: X3 değişkeninin en az bir düzeyi diğerlerinden farklı Y değerlerine sahiptir.")
print(f"F değeri: {f_value:.4f}")
print(f"p değeri: {p_value:.4f}")
print(f"Alfa değeri: {alpha}")

if p_value < alpha:
    print("Sonuç: H₀ hipotezi reddedilir. X3 değişkeninin Y üzerinde istatistiksel olarak anlamlı bir etkisi vardır.")
else:
    print("Sonuç: H₀ hipotezi reddedilemez. X3 değişkeninin Y üzerinde istatistiksel olarak anlamlı bir etkisi yoktur.")

# Grup ortalamaları
print("\nİlaç Gruplarına Göre Y Değerlerinin Ortalamaları:")
print(df.groupby('X3')['Y'].mean())

# X3'ün R-kare değeri
print(f"\nR-kare değeri: {model_x3.rsquared:.4f}")
print(f"Düzeltilmiş R-kare değeri: {model_x3.rsquared_adj:.4f}")