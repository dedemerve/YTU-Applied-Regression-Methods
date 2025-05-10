import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Veri setini oluşturma
data = {
    'X1': [2, 4, 9, 11, 3, 5, 5, 10, 4, 7, 6, 1],
    'X2': [9, 2, -8, -10, 6, -6, 0, -12, 4, -2, -4, 8],
    'X3': ['A', 'A', 'C', 'C', 'B', 'B', 'C', 'C', 'A', 'C', 'B', 'B'],
    'Y': [7, 8, 1, 0, 5, 3, 2, -4, 10, -3, 5, 6]
}

# DataFrame oluşturma
df = pd.DataFrame(data)

# Referans kategori kullanmadan model formülasyonu
# patsy formülü: dummy değişken kullanarak ancak referans kategori oluşturmadan
# 0+C(X3) ifadesi, sabit terim kullanmadan tüm kategorileri dahil eder
formula = "Y ~ X1 + X2 + 0 + C(X3, Treatment)"
y, X = patsy.dmatrices(formula, df, return_type='dataframe')

# Model tahmini
model = sm.OLS(y, X).fit()

print("Model Özeti (X3 için referans kategori kullanmadan):")
print(model.summary())

# Sabit terimi manuel olarak eklemek için
X_with_const = sm.add_constant(X)
model_with_const = sm.OLS(y, X_with_const).fit()

print("\nModel Özeti (Manuel sabit terim eklenmiş):")
print(model_with_const.summary())

# Multiplikatif kimlik kısıtı (sum-to-zero) ile model tahmin etme (alternatif yöntem)
formula_sum_to_zero = "Y ~ X1 + X2 + C(X3, Sum)"
y_sum, X_sum = patsy.dmatrices(formula_sum_to_zero, df, return_type='dataframe')

model_sum = sm.OLS(y_sum, X_sum).fit()

print("\nModel Özeti (Sum-to-zero kısıtı ile):")
print(model_sum.summary())

# Her kategorinin sıklığını hesaplama
print("\nKategori Frekansları:")
print(df['X3'].value_counts())

# Her kategorinin ortalama Y değeri
print("\nKategorilere Göre Y Ortalamaları:")
print(df.groupby('X3')['Y'].mean())