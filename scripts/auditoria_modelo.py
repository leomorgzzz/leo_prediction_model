import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    rf_model = joblib.load('../binarios/modelo_clinvar_v2.pkl')
    vectorizer = joblib.load('../binarios/vectorizador_clinvar_v2.pkl')
    print("Modelos Cargados [OK]")
except:
    print("Error: No encuentro los .pkl en ../binarios/")
    exit()

importancias = rf_model.feature_importances_
nombres_features = rf_model.feature_names_in_

df_imp = pd.DataFrame({'Feature': nombres_features, 'Importance': importancias})
df_imp = df_imp.sort_values(by='Importance', ascending=False).head(50) 

plt.figure(figsize=(12, 8))
sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis')

plt.title('Qué busca el modelo?', fontdict={'family': 'monospace', 'size': 16})
plt.xlabel('Peso en decisión', fontdict={'family': 'monospace'})
plt.ylabel('Variable / Palabra', fontdict={'family': 'monospace'})
plt.grid(axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('auditoria_modelo.png')
print("grafico generado: 'auditoria_modelo.png' [OK]")
plt.show()