import pandas as pd
url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
print("Buscando misterios médicos (Variantes de Significado Incierto)...")


chunk_size = 100000
uncertain_variants = []


for chunk in pd.read_csv(url, compression='gzip', sep='\t', chunksize=chunk_size):
    filtered = chunk[chunk['ClinicalSignificance'] == 'Uncertain significance']
    
   
    cols = ['PhenotypeList', 'Type', 'GeneSymbol', 'ClinicalSignificance', 'RCVaccession']
    
    available_cols = [c for c in cols if c in filtered.columns]
    
    uncertain_variants.append(filtered[available_cols])
    if sum([len(x) for x in uncertain_variants]) > 500:
        break

df_uncertain = pd.concat(uncertain_variants).head(500)

df_uncertain.to_csv('misterios_clinicos.csv', index=False)
print(f"¡Listo! Se generó 'misterios_clinicos.csv' con {len(df_uncertain)} variantes desconocidas.")
print("Sube este archivo a tu Shiny App y mira qué opina la IA.")
