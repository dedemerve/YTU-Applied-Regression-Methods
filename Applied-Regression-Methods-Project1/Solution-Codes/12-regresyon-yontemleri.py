import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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

# Beta katsayıları
beta_1 = model.params.iloc[1]  # X1 katsayısı
beta_2 = model.params.iloc[2]  # X2 katsayısı

# Varyans-kovaryans matrisi
vcov = model.cov_params()
var_beta_1 = vcov.iloc[1, 1]  # β₁'in varyansı
var_beta_2 = vcov.iloc[2, 2]  # β₂'nin varyansı
cov_beta1_beta2 = vcov.iloc[1, 2]  # β₁ ve β₂ arasındaki kovaryans

# %95 ortak güven bölgesi için F değeri
alpha = 0.05
p = 2  # İlgilenilen parametre sayısı (β₁ ve β₂)
df_residual = model.df_resid  # Hata serbestlik derecesi
F_critical = stats.f.ppf(1-alpha, p, df_residual)

print("(β₁, β₂) %95 Ortak Güven Bölgesi:")
print("---------------------------------------------------")
print(f"β₁ katsayısı: {beta_1:.4f}")
print(f"β₂ katsayısı: {beta_2:.4f}")
print(f"Var(β₁): {var_beta_1:.6f}")
print(f"Var(β₂): {var_beta_2:.6f}")
print(f"Cov(β₁, β₂): {cov_beta1_beta2:.6f}")
print(f"Kritik F değeri (α=0.05, df=({p}, {df_residual})): {F_critical:.4f}")

# Ortak güven bölgesi elipsini çizme fonksiyonu
def confidence_ellipse(x_center, y_center, cov_matrix, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Kovaryans matrisinden güven elipsini çizer.
    
    Parametreler:
    x_center, y_center : Elipsin merkezi
    cov_matrix : 2x2 kovaryans matrisi
    ax : Matplotlib ekseni
    n_std : Standart sapma sayısı (güven aralığına göre ayarlanır)
    """
    pearson = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
    # Kovaryans matrisinin özdeğerlerini ve özvektörlerini hesapla
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    
    # Elipsin yarı eksenleri
    width, height = 2 * n_std * np.sqrt(eigvals)
    
    # Elipsin dönüş açısı
    theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    
    # Elips nesnesi oluştur
    ellipse = Ellipse((0, 0), width=width, height=height, angle=theta, facecolor=facecolor, **kwargs)
    
    # Koordinat dönüşümü
    transf = transforms.Affine2D().translate(x_center, y_center)
    ellipse.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse)

# Elipsin çizimi için gerekli varyans-kovaryans matrisi (sadece β₁ ve β₂ için)
cov_submatrix = np.array([[var_beta_1, cov_beta1_beta2], [cov_beta1_beta2, var_beta_2]])

# n_std için, F_critical değerini kullanarak elips genişliğini hesapla
# F dağılımından güven elipsinin ölçeğini hesapla
ellipse_scale = np.sqrt(p * F_critical)

# Güven elipsini çizme
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)

# Nokta tahmini noktası
plt.scatter(beta_1, beta_2, color='red', s=100, label='$(\hat{\\beta}_1, \hat{\\beta}_2)$')

# %95 güven elipsi
confidence_ellipse(beta_1, beta_2, cov_submatrix, ax, n_std=ellipse_scale, edgecolor='blue', label='%95 Güven Bölgesi')

# Referans çizgileri
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)

# Eksen isimleri ve başlık
plt.xlabel('$\\beta_1$')
plt.ylabel('$\\beta_2$')
plt.title('$(\\beta_1, \\beta_2)$ için %95 Ortak Güven Bölgesi')
plt.grid(True, alpha=0.3)
plt.legend()

# Elipsin uç noktalarını hesaplama
eigvals, eigvecs = np.linalg.eigh(cov_submatrix)
width, height = 2 * ellipse_scale * np.sqrt(eigvals)

# X ve Y sınırlarını ayarlama
x_min = beta_1 - width/1.5
x_max = beta_1 + width/1.5
y_min = beta_2 - height/1.5
y_max = beta_2 + height/1.5

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Elipsin denklemi
print("\nOrtak Güven Bölgesi Denklemi:")
print(f"({F_critical} * p) * (x - {beta_1})² / {var_beta_1} + ({F_critical} * p) * (y - {beta_2})² / {var_beta_2} - 2 * ({F_critical} * p) * (x - {beta_1})(y - {beta_2}) * {cov_beta1_beta2} / ({var_beta_1} * {var_beta_2}) ≤ 1")

plt.show()