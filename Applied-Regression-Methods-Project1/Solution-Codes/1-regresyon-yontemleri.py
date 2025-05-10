import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Veri setini oluşturalım
data = {
    'Gozlem': range(1, 13),
    'Y': [7, 8, 1, 0, 5, 3, 2, -4, 10, -3, 5, 6],
    'X1': [2, 4, 9, 11, 3, 5, 5, 10, 4, 7, 6, 1],
    'X2': [9, 2, -8, -10, 6, -6, 0, -12, 4, -2, -4, 8],
    'X3': ['A', 'A', 'C', 'C', 'B', 'B', 'C', 'C', 'A', 'C', 'B', 'B']
}

df = pd.DataFrame(data)

# Formula API kullanarak kategorik değişkeni (X3) otomatik olarak işleyelim
# C() fonksiyonu statsmodels'e X3'ün kategorik bir değişken olduğunu söyler
formula = 'Y ~ X1 + X2 + C(X3, Treatment(reference="C"))'
model = smf.ols(formula=formula, data=df).fit()

# Sonuçları yazdıralım
print(model.summary())

print("\nTahmin edilen regresyon katsayıları:")
print(f"β₀ (Sabit): {model.params[0]:.4f}")
print(f"β₁ (X1 katsayısı): {model.params[1]:.4f}")
print(f"β₂ (X2 katsayısı): {model.params[2]:.4f}")
print(f"β₃ (X3_A katsayısı): {model.params[3]:.4f}")  # C(X3)[T.A]
print(f"β₄ (X3_B katsayısı): {model.params[4]:.4f}")  # C(X3)[T.B]

# Sadece referansla gerçek parametre isimlerini gösterelim
print("\nGerçek parametre isimleri:")
for i, name in enumerate(model.params.index):
    print(f"β{i}: {name}")