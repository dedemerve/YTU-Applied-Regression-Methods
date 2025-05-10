import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time

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
X = sm.add_constant(X)
model_full = sm.OLS(y, X).fit()

# Orijinal modelin R² değeri
original_r2 = model_full.rsquared
print(f"Orijinal modelin R² değeri: {original_r2:.4f}")

# Permütasyon testi için fonksiyon
def permutation_test(X, y, n_permutations=1000, random_seed=42):
    np.random.seed(random_seed)
    
    # Orijinal model ve R²
    model = sm.OLS(y, X).fit()
    original_r2 = model.rsquared
    
    # Permütasyon R² değerlerini saklayacak liste
    permuted_r2_values = []
    
    # Zaman ölçümü başlat
    start_time = time.time()
    
    # n_permutations sayıda permütasyon yaparak R² değerlerini hesapla
    for i in range(n_permutations):
        # Y değerlerini karıştır (permütasyon)
        y_permuted = np.random.permutation(y)
        
        # Karıştırılmış Y değerleriyle model kur ve R² hesapla
        model_permuted = sm.OLS(y_permuted, X).fit()
        permuted_r2 = model_permuted.rsquared
        
        # R² değerini listeye ekle
        permuted_r2_values.append(permuted_r2)
    
    # Zaman ölçümü bitir
    end_time = time.time()
    
    # p-değeri hesapla: Orijinal R²'den büyük veya eşit R² değerlerinin oranı
    p_value = np.mean(np.array(permuted_r2_values) >= original_r2)
    
    return permuted_r2_values, p_value, end_time - start_time

# Permütasyon testini gerçekleştir
n_permutations = 1000
permuted_r2_values, p_value, elapsed_time = permutation_test(X, y, n_permutations)

print(f"Permütasyon testi sonuçları (n_permutations = {n_permutations}):")
print(f"Orijinal modelin R² değeri: {original_r2:.4f}")
print(f"Permütasyon p-değeri: {p_value:.4f}")
print(f"İşlem süresi: {elapsed_time:.2f} saniye")

# Permütasyon dağılımının histogramını çiz
plt.figure(figsize=(10, 6))
plt.hist(permuted_r2_values, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(x=original_r2, color='red', linestyle='--', linewidth=2, label=f'Orijinal R² = {original_r2:.4f}')
plt.title('Permütasyon Testi R² Dağılımı')
plt.xlabel('R² Değeri')
plt.ylabel('Frekans')
plt.legend()
plt.grid(True, alpha=0.3)

# Alternatif permütasyon testi: F istatistiği
def permutation_f_test(X, y, n_permutations=1000, random_seed=42):
    np.random.seed(random_seed)
    
    # Orijinal model ve F istatistiği
    model = sm.OLS(y, X).fit()
    original_f = model.fvalue
    
    # Permütasyon F değerlerini saklayacak liste
    permuted_f_values = []
    
    # n_permutations sayıda permütasyon yaparak F değerlerini hesapla
    for i in range(n_permutations):
        # Y değerlerini karıştır (permütasyon)
        y_permuted = np.random.permutation(y)
        
        # Karıştırılmış Y değerleriyle model kur ve F hesapla
        model_permuted = sm.OLS(y_permuted, X).fit()
        permuted_f = model_permuted.fvalue
        
        # F değerini listeye ekle
        permuted_f_values.append(permuted_f)
    
    # p-değeri hesapla: Orijinal F'den büyük veya eşit F değerlerinin oranı
    p_value = np.mean(np.array(permuted_f_values) >= original_f)
    
    return permuted_f_values, p_value

# F istatistiği permütasyon testini gerçekleştir
permuted_f_values, p_value_f = permutation_f_test(X, y, n_permutations)

print(f"\nF istatistiği permütasyon testi sonuçları:")
print(f"Orijinal modelin F değeri: {model_full.fvalue:.4f}")
print(f"Permütasyon p-değeri (F): {p_value_f:.4f}")

# F dağılımının histogramını çiz
plt.figure(figsize=(10, 6))
plt.hist(permuted_f_values, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(x=model_full.fvalue, color='red', linestyle='--', linewidth=2, label=f'Orijinal F = {model_full.fvalue:.4f}')
plt.title('Permütasyon Testi F İstatistiği Dağılımı')
plt.xlabel('F Değeri')
plt.ylabel('Frekans')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()