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

# Referans kategori kullanmadan ve etkileşim etkileri dahil edilerek model formülasyonu
# Patsy'nin ürettiği kolon isimleri için model.params.index'i kontrol edeceğiz
formula = """Y ~ X1 + X2 + C(X3, Treatment) + 
             X1:C(X3, Treatment) + X2:C(X3, Treatment) - 1"""

y, X = patsy.dmatrices(formula, df, return_type='dataframe')

# Modeli tahmin et
model = sm.OLS(y, X).fit()

print("Etkileşim Etkileri Dahil Model Özeti (Referans kategori kullanmadan):")
print(model.summary())

# Model parametrelerinin isimlerini kontrol edelim
print("\nModel Parametre İsimleri:")
for i, param_name in enumerate(model.params.index):
    print(f"{i}: {param_name}")

# Etkileşim terimlerinin görselleştirilmesi
# X1 ve X3 etkileşimi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for category in df['X3'].unique():
    subset = df[df['X3'] == category]
    plt.scatter(subset['X1'], subset['Y'], label=f'X3 = {category}')
    
    # Kategori için regresyon çizgisi
    if len(subset) > 1:  # En az 2 nokta varsa regresyon çizgisi çiz
        slope = np.polyfit(subset['X1'], subset['Y'], 1)[0]
        x_range = np.linspace(df['X1'].min(), df['X1'].max(), 100)
        plt.plot(x_range, slope * (x_range - df['X1'].min()) + subset['Y'].mean())
        
plt.xlabel('X1')
plt.ylabel('Y')
plt.title('X1 ve X3 Etkileşimi')
plt.legend()
plt.grid(alpha=0.3)

# X2 ve X3 etkileşimi
plt.subplot(1, 2, 2)
for category in df['X3'].unique():
    subset = df[df['X3'] == category]
    plt.scatter(subset['X2'], subset['Y'], label=f'X3 = {category}')
    
    # Kategori için regresyon çizgisi
    if len(subset) > 1:  # En az 2 nokta varsa regresyon çizgisi çiz
        slope = np.polyfit(subset['X2'], subset['Y'], 1)[0]
        x_range = np.linspace(df['X2'].min(), df['X2'].max(), 100)
        plt.plot(x_range, slope * (x_range - df['X2'].min()) + subset['Y'].mean())
        
plt.xlabel('X2')
plt.ylabel('Y')
plt.title('X2 ve X3 Etkileşimi')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Her bir kategorideki X1 ve X2'nin etkilerini hesaplama
# Doğru parametre isimlerini kullanarak
print("\nX3 Kategorilerine Göre X1 ve X2'nin Marjinal Etkileri:")
print("------------------------------------------------------------")

# A kategorisi için etkiler
X1_A_effect = model.params['X1']
X2_A_effect = model.params['X2']

# B kategorisi için etkiler
X1_B_effect = model.params['X1'] + model.params['X1:C(X3, Treatment)[T.B]']
X2_B_effect = model.params['X2'] + model.params['X2:C(X3, Treatment)[T.B]']

# C kategorisi için etkiler
X1_C_effect = model.params['X1'] + model.params['X1:C(X3, Treatment)[T.C]']
X2_C_effect = model.params['X2'] + model.params['X2:C(X3, Treatment)[T.C]']

print("X3 = A:")
print(f"X1'in etkisi: {X1_A_effect:.4f}")
print(f"X2'nin etkisi: {X2_A_effect:.4f}")

print("\nX3 = B:")
print(f"X1'in etkisi: {X1_B_effect:.4f}")
print(f"X2'nin etkisi: {X2_B_effect:.4f}")

print("\nX3 = C:")
print(f"X1'in etkisi: {X1_C_effect:.4f}")
print(f"X2'nin etkisi: {X2_C_effect:.4f}")

# Karşılaştırma için etkileşimsiz model
formula_no_interaction = "Y ~ X1 + X2 + C(X3, Treatment) - 1"
y_no_int, X_no_int = patsy.dmatrices(formula_no_interaction, df, return_type='dataframe')
model_no_int = sm.OLS(y_no_int, X_no_int).fit()

print("\nModel Karşılaştırması:")
print(f"Etkileşimsiz Model R²: {model_no_int.rsquared:.4f}")
print(f"Etkileşimli Model R²: {model.rsquared:.4f}")
print(f"Etkileşimsiz Model Düzeltilmiş R²: {model_no_int.rsquared_adj:.4f}")
print(f"Etkileşimli Model Düzeltilmiş R²: {model.rsquared_adj:.4f}")

# F testi ile model karşılaştırması
from statsmodels.stats.anova import anova_lm

try:
    anova_table = anova_lm(model_no_int, model)
    print("\nANOVA Model Karşılaştırması:")
    print(anova_table)
except Exception as e:
    print(f"\nANOVA karşılaştırması yapılamadı: {e}")
    
    # Manuel F testi hesaplama
    n = len(df)  # Gözlem sayısı
    k1 = len(model_no_int.params)  # Kısıtlı model parametre sayısı
    k2 = len(model.params)  # Tam model parametre sayısı
    df1 = k2 - k1  # Serbestlik derecesi farkı
    df2 = n - k2  # Tam model hata serbestlik derecesi
    
    # F istatistiği hesaplama
    SSE1 = sum(model_no_int.resid**2)  # Kısıtlı model hata kareler toplamı
    SSE2 = sum(model.resid**2)  # Tam model hata kareler toplamı
    F_stat = ((SSE1 - SSE2) / df1) / (SSE2 / df2)
    
    # p-değeri hesaplama
    from scipy import stats
    p_value = 1 - stats.f.cdf(F_stat, df1, df2)
    
    print("\nManuel F Testi:")
    print(f"F İstatistiği: {F_stat:.4f}")
    print(f"Serbestlik Dereceleri: ({df1}, {df2})")
    print(f"p-değeri: {p_value:.4f}")