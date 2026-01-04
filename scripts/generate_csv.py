import pandas as pd

url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
print("Descargando datos de prueba desconocidos...")

df_test = pd.read_csv(url, compression='gzip', sep='\t', 
                      skiprows=range(1, 500000), nrows=50) 

cols_to_keep = ['PhenotypeList', 'Type', 'GeneSymbol', 'ClinicalSignificance']
df_test = df_test[cols_to_keep]


df_test.to_csv('datos_reales_prueba.csv', index=False)

print("Archivo 'datos_reales_prueba.csv' generado.")
print("Este archivo contiene variantes reales que tu modelo NO conoce.")
