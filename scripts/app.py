from shiny import App, render, ui, reactive
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("inicializando aplicación...")

try:
    rf_model = joblib.load('modelo_clinvar_v2.pkl')
    vectorizer = joblib.load('vectorizador_clinvar_v2.pkl')
    print("Modelos cargados exitosamente.")
except Exception as e:
    print(f"Error: No encuentro los archivos .pkl.\n{e}")


app_ui = ui.page_fluid(
    ui.tags.style("""
        .card { box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); transition: 0.3s; }
        .h2 { color: #2c3e50; }
    """),

    ui.h2("Proyecto: Predictor de Patogenicidad"),
    ui.markdown("Estructura: | **Backend:** Random Forest + NLP | **Frontend:** Shiny for Python"),
    ui.hr(),
    
    ui.layout_sidebar(
        # 1. La barra lateral
        ui.sidebar(
            ui.h4("+ Cargar Datos"),
            ui.input_file("file_upload", "Sube tu archivo CSV", accept=[".csv"], multiple=False),
            ui.download_button("download_pred", "Descargar Reporte", class_="btn-success"),
            ui.hr(),
            ui.h5("Requisitos del CSV:"),
            ui.markdown("""
            Tu tabla debe tener columnas parecidas a ClinVar:
            * `PhenotypeList` (Texto médico)
            * `Type` (Ej: 'Deletion', 'single nucleotide variant')
            """),
            ui.input_action_button("btn_example", "Cargar Ejemplo de Prueba", class_="btn-secondary btn-sm")
        ),
        
        # 2. El contenido principal
        ui.navset_tab(
            # Pestaña 1: Tabla
            ui.nav_panel("Resultados", 
                   ui.output_text_verbatim("status_text"),
                   ui.output_data_frame("prediction_table")
            ),
            # Pestaña 2: Gráficos
            ui.nav_panel("Distribución", 
                   ui.output_plot("dist_plot")
            ),
            # Pestaña 3: Explicación
            ui.nav_panel("¿Cómo funciona?",
                   ui.markdown("""
                   **Modelo:** Random Forest Classifier (Entrenado con ~200k variantes).
                   **Lógica:** Analiza palabras clave en el fenotipo (ej: 'syndrome', 'congenital') 
                   y el tipo de mutación (ej: 'Deletion').
                   """)
            )
        )
    )
)


def server(input, output, session):

    val = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.file_upload)
    def load_csv():
        file_info = input.file_upload()
        if not file_info: return
        df = pd.read_csv(file_info[0]["datapath"])
        val.set(process_dataframe(df))

    @reactive.Effect
    @reactive.event(input.btn_example)
    def load_example():
        # Ejemplo falso para probar
        data = {
            'PhenotypeList': [
                'Usher Syndrome type 1 congenital', 
                'Benign polymorphism observed in population',
                'Hereditary cardiomyopathy and heart failure',
                'Not specified provided by submitter'
            ],
            'Type': [
                'Deletion', 
                'single nucleotide variant', 
                'Duplication', 
                'single nucleotide variant'
            ]
        }
        val.set(process_dataframe(pd.DataFrame(data)))

    def process_dataframe(df_input):
        # 1. Limpieza básica
        if 'PhenotypeList' not in df_input.columns:
            df_input['PhenotypeList'] = "not specified"
        df_input['PhenotypeList'] = df_input['PhenotypeList'].fillna('')
        
        # 2. NLP
        X_text = vectorizer.transform(df_input['PhenotypeList'])
        text_features = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())
        
        # 3. Estructura
        if 'Type' not in df_input.columns:
            df_input['Type'] = 'single nucleotide variant'
            
        type_dummies = pd.get_dummies(df_input['Type'], prefix='type')
        
        # 4. UNIÓN
        X_final = pd.concat([text_features, type_dummies], axis=1)
        expected_cols = rf_model.feature_names_in_
        X_final = X_final.reindex(columns=expected_cols, fill_value=0)
        
        # 5. Predicción
        preds = rf_model.predict(X_final)
        probs = rf_model.predict_proba(X_final)[:, 1]
        
        df_input['Predicción'] = ["X -> PATOGÉNICA" if p == 1 else "O -> BENIGNA" for p in preds]
        df_input['Confianza (%)'] = (probs * 100).round(1)
        
        return df_input

    # SALIDAS
    @output
    @render.text
    def status_text():
        if val.get() is None: return "Esperando archivo..."
        return f"Procesadas {len(val.get())} variantes."

    @output
    @render.data_frame
    def prediction_table():
        df = val.get()
        if df is None: return pd.DataFrame()
        cols = ['Predicción', 'Confianza (%)', 'Type', 'PhenotypeList']
        return df[cols]
    
    @render.download(filename="predicciones_genomicas.csv")
    def download_pred():
        df = val.get()
        if df is not None:
            yield df.to_csv(index=False)

    @output
    @render.plot
    def dist_plot():
        df = val.get()
        if df is None: return None
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df, x='Confianza (%)', hue='Predicción', 
                     multiple="stack", palette={'X -> PATOGÉNICA': '#e74c3c', 'O -> BENIGNA': '#2ecc71'}, ax=ax)
        ax.set_title("Distribución de Confianza")
        ax.set_xlim(0, 100)
        return fig

app = App(app_ui, server)