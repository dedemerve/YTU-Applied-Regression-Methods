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

# Bağımsız değişkenler
X = df[['X1', 'X2', 'X3_A', 'X3_B']]
X = sm.add_constant(X)  # Sabit terim ekleme

# Bağımlı değişken
y = df['Y']

# Çoklu doğrusal regresyon modeli
model = sm.OLS(y, X).fit()

# Varyans-kovaryans matrisi
vcov_matrix = model.cov_params()
print("Varyans-Kovaryans Matrisi:")
print(vcov_matrix)

# β₁ (X1) ve β₂ (X2) arasındaki kovaryans değeri
# Not: X = [const, X1, X2, X3_A, X3_B]
# Bu nedenle, β₁ indeksi 1, β₂ indeksi 2'dir
cov_beta1_beta2 = vcov_matrix.iloc[1, 2]
print(f"\nCov(β₁, β₂) = {cov_beta1_beta2:.6f}")