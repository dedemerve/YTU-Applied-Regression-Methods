import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini oluşturma
data = {
    'X1': [2, 4, 9, 11, 3, 5, 5, 10, 4, 7, 6, 1],
    'X2': [9, 2, -8, -10, 6, -6, 0, -12, 4, -2, -4, 8],
    'X3': ['A', 'A', 'C', 'C', 'B', 'B', 'C', 'C', 'A', 'C', 'B', 'B'],
    'Y': [7, 8, 1, 0, 5, 3, 2, -4, 10, -3, 5, 6]
}

# DataFrame oluşturma
df = pd.DataFrame(data)

# 16. SORU: Referans kategori kullanmadan ve etkileşim etkileri olmadan model
formula_16 = "Y ~ X1 + X2 + C(X3, Treatment) - 1"
y_16, X_16 = patsy.dmatrices(formula_16, df, return_type='dataframe')
model_16 = sm.OLS(y_16, X_16).fit()

# 17. SORU: Referans kategori kullanmadan ve etkileşim etkileri dahil ederek model
formula_17 = """Y ~ X1 + X2 + C(X3, Treatment) + 
             X1:C(X3, Treatment) + X2:C(X3, Treatment) - 1"""
y_17, X_17 = patsy.dmatrices(formula_17, df, return_type='dataframe')
model_17 = sm.OLS(y_17, X_17).fit()

# 16. Modelin varyans-kovaryans matrisi
vcov_16 = model_16.cov_params()

# 17. Modelin varyans-kovaryans matrisi
vcov_17 = model_17.cov_params()

# Varyans-kovaryans matrislerini inceleme
print("16. Model (Etkileşimsiz) için Varyans-Kovaryans Matrisi:")
print(vcov_16)
print("\nBoyut:", vcov_16.shape)
print("\nKöşegen elemanlar (varyanslar):")
for i, param in enumerate(vcov_16.index):
    print(f"{param}: {vcov_16.iloc[i, i]:.6f}")

print("\n\n17. Model (Etkileşimli) için Varyans-Kovaryans Matrisi:")
print(vcov_17)
print("\nBoyut:", vcov_17.shape)
print("\nKöşegen elemanlar (varyanslar):")
for i, param in enumerate(vcov_17.index):
    print(f"{param}: {vcov_17.iloc[i, i]:.6f}")

# Varyans-kovaryans matrislerinin görselleştirilmesi
plt.figure(figsize=(14, 6))

# 16. Modelin varyans-kovaryans matrisinin görselleştirilmesi
plt.subplot(1, 2, 1)
sns.heatmap(vcov_16, annot=False, cmap='viridis', fmt='.2e', 
            xticklabels=vcov_16.index, yticklabels=vcov_16.index)
plt.title('16. Model (Etkileşimsiz) için \nVaryans-Kovaryans Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# 17. Modelin varyans-kovaryans matrisinin görselleştirilmesi
plt.subplot(1, 2, 2)
sns.heatmap(vcov_17, annot=False, cmap='viridis', fmt='.2e', 
            xticklabels=vcov_17.index, yticklabels=vcov_17.index)
plt.title('17. Model (Etkileşimli) için \nVaryans-Kovaryans Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('vcov_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Korelasyon matrislerine dönüştürme
def cov_to_corr(cov_matrix):
    """Kovaryans matrisini korelasyon matrisine dönüştürür"""
    var = np.sqrt(np.diag(cov_matrix))
    outer_var = np.outer(var, var)
    corr = cov_matrix / outer_var
    corr[cov_matrix == 0] = 0  # Sıfır kovaryanslar için sıfır korelasyon
    return corr

# Kovaryans matrislerini korelasyon matrislerine dönüştürme
corr_16 = cov_to_corr(vcov_16.values)
corr_17 = cov_to_corr(vcov_17.values)

# Korelasyon matrislerini görselleştirme
plt.figure(figsize=(14, 6))

# 16. Modelin korelasyon matrisinin görselleştirilmesi
plt.subplot(1, 2, 1)
sns.heatmap(pd.DataFrame(corr_16, index=vcov_16.index, columns=vcov_16.columns), 
            annot=False, cmap='coolwarm', vmin=-1, vmax=1, 
            xticklabels=vcov_16.index, yticklabels=vcov_16.index)
plt.title('16. Model (Etkileşimsiz) için \nKorelasyon Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# 17. Modelin korelasyon matrisinin görselleştirilmesi
plt.subplot(1, 2, 2)
sns.heatmap(pd.DataFrame(corr_17, index=vcov_17.index, columns=vcov_17.columns), 
            annot=False, cmap='coolwarm', vmin=-1, vmax=1, 
            xticklabels=vcov_17.index, yticklabels=vcov_17.index)
plt.title('17. Model (Etkileşimli) için \nKorelasyon Matrisi')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('correlation_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Varyans Şişme Faktörlerini (VIF) hesaplama ve karşılaştırma
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 16. Model için VIF hesaplama
vif_16 = pd.DataFrame()
vif_16["Variable"] = X_16.columns
vif_16["VIF"] = [variance_inflation_factor(X_16.values, i) for i in range(X_16.shape[1])]

# 17. Model için VIF hesaplama (Uyarı: Etkileşim terimleri ile VIF yüksek çıkabilir)
vif_17 = pd.DataFrame()
vif_17["Variable"] = X_17.columns
vif_17["VIF"] = [variance_inflation_factor(X_17.values, i) for i in range(X_17.shape[1])]

print("\n16. Model için Varyans Şişme Faktörleri (VIF):")
print(vif_16)

print("\n17. Model için Varyans Şişme Faktörleri (VIF):")
print(vif_17)

# Koşul numaralarını karşılaştırma
print("\nKoşul Sayıları:")
print(f"16. Model: {np.linalg.cond(X_16.values):.2f}")
print(f"17. Model: {np.linalg.cond(X_17.values):.2f}")

# Özdeğerlerin hesaplanması ve karşılaştırılması
eigenvalues_16 = np.linalg.eigvals(X_16.T @ X_16)
eigenvalues_17 = np.linalg.eigvals(X_17.T @ X_17)

print("\n16. Model için Özdeğerler:")
for i, val in enumerate(sorted(eigenvalues_16)):
    print(f"λ{i+1}: {val:.6f}")

print("\n17. Model için Özdeğerler:")
for i, val in enumerate(sorted(eigenvalues_17)):
    print(f"λ{i+1}: {val:.6f}")

# Özdeğerlerin oranı (en büyük / en küçük)
print("\nÖzdeğer Oranları (Koşul Sayısı):")
print(f"16. Model: {max(eigenvalues_16) / min(eigenvalues_16):.2f}")
print(f"17. Model: {max(eigenvalues_17) / min(eigenvalues_17):.2f}")