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

# Model özeti
print("Model Özeti:")
print(model.summary())

# X₁=3, X₂=5, X₃=B için veri noktası
x_pred = pd.DataFrame({
    'const': [1],
    'X1': [3],
    'X2': [5],
    'X3_A': [0],  # X3 = B olduğu için X3_A = 0
    'X3_B': [1]   # X3 = B olduğu için X3_B = 1
})

# Tahmin değeri
y_pred = model.predict(x_pred)
print(f"\nX₁=3, X₂=5, X₃=B için tahmin değeri: {y_pred[0]:.4f}")

# %95 güven aralığı 
# Not: Bu ortalama yanıt için güven aralığıdır (tek bir tahmin için değil)
alpha = 0.05
t_critical = stats.t.ppf(1 - alpha/2, model.df_resid)

# Modelin varyans-kovaryans matrisi
vcov = model.cov_params()

# Tahmin varyansı: Var(ŷ) = x' * Var(β) * x
x_array = x_pred.values
se_pred = np.sqrt(x_array @ vcov.values @ x_array.T)[0, 0]

# Güven aralığı sınırları
lower_bound = y_pred[0] - t_critical * se_pred
upper_bound = y_pred[0] + t_critical * se_pred

print(f"%95 güven aralığı: [{lower_bound:.4f}, {upper_bound:.4f}]")

# Alternatif yöntem - statsmodels'in get_prediction metodu
prediction = model.get_prediction(x_pred)
conf_interval = prediction.conf_int(alpha=0.05)

print("\nStatsmodels get_prediction() kullanarak:")
print(f"Tahmin: {prediction.predicted_mean[0]:.4f}")
print(f"%95 güven aralığı: [{conf_interval[0, 0]:.4f}, {conf_interval[0, 1]:.4f}]")