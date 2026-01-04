import os 
from shiny import App, render, ui, reactive
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from shinywidgets import output_widget, render_widget  

print("INICIANDO APP...")

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'binarios', 'modelo_clinvar_v2.pkl')
vec_path = os.path.join(base_dir, '..', 'binarios', 'vectorizador_clinvar_v2.pkl')

try:    
    rf_model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    print("MODELOS CARGADOS CORRECTAMENTE [OK]")
except Exception as e:
    print(f"ERROR CARGANDO MODELOS [FAIL]\n{e}")

#ui
app_ui = ui.page_fluid(
    ui.tags.style("""
        /* typo */
        body, h2, h4, h5, .btn, .table, .shiny-input-container { 
            font-family: 'Courier New', Courier, monospace !important; 
        }
        .container-fluid { padding: 40px !important; }
        
        /* style */
        .form-control { border-radius: 0px; border: 1px solid #333; }
        .btn { border-radius: 0px; font-weight: bold; border: 1px solid #333; }
        .h2 { font-weight: 900; letter-spacing: -1px; margin-bottom: 20px; }
        .card { border: 1px solid #333; box-shadow: 5px 5px 0px #333; }
        
        /* bar */
        .progress { height: 25px !important; border-radius: 0px !important; border: 1px solid #333; background-color: #eee; }
        .progress-bar { background-color: #2c3e50 !important; font-size: 14px; line-height: 25px; color: #fff; }
    """),

    #header
    ui.h2(">_ PROYECTO SHINY: // Modelo de Predicción"),
    ui.markdown("`v1.0`"),
    ui.hr(),
    
    ui.layout_sidebar(
        #sidebar
        ui.sidebar(
            ui.h4("[+] Cargar Datos"),
            
            ui.input_file("file_upload", "Seleccione Archivo:", 
                          accept=[".csv", ".txt", ".tsv"], 
                          button_label="[ BROWSE ]", 
                          placeholder="NO_FILE_SELECTED"),
            
            ui.download_button("download_pred", "[v] Descargar Resultados", class_="btn-success"),
            ui.hr(),
            
            ui.h5("Archivos Utilizables:"),
            ui.markdown("""
            * **Extensión:** CSV / TXT / TSV
            * **Columnas Necesarias:**
              - `PhenotypeList`
              - `Type`
            """),
            ui.br(),
            ui.input_action_button("btn_example", "[>] Cargar Ejemplo", class_="btn-secondary btn-sm"),
            
            width=350 
        ),
        
        #pestañas
        ui.navset_tab(
            #pestaña 1
            ui.nav_panel("[1] Datos Cargados", 
                   ui.output_text_verbatim("status_text"),
                   ui.output_data_frame("prediction_table")
            ),
            #pestaña 2
            ui.nav_panel("[2] Visualizar Distribución", 
                   output_widget("interactive_plot") 
            ),
            #pestaña 3
            ui.nav_panel("[3] Manual",
                   ui.markdown("""
                   #### **MANUAL DE USUARIO**
                   
                   **INTERACTIVIDAD**
                   * **Puntos:** Pasa el cursor para ver detalles.
                   * **Zoom:** Selecciona un área para acercar.
                   * **Doble Click:** Restablecer vista.
                   
                   **OUTPUT CODES**
                   * `[!] PATOGÉNICO` : Riesgo Alto (>70%)
                   * `[?] INCIERTO`   : Riesgo Incierto (30-70%)
                   * `[OK] BENIGNO`   : Riesgo Bajo (<30%)
                
                   *Creador*
                    Leonardo Morales Rodríguez
                   """)
            )
        )
    )
)

#server
def server(input, output, session):

    val = reactive.Value(None)
    msg_error = reactive.Value("") 

    def smart_load(filepath, filename):
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_csv(filepath, sep='\t')

            column_mapping = {
                'PhenotypeList': ['Condition(s)', 'Conditions', 'Phenotypes', 'PhenotypeIDS', 'Name'],
                'Type': ['Variant type', 'Variation Type', 'class_type']
            }

            cols_found = df.columns.tolist()
            rename_dict = {}

            for target, synonyms in column_mapping.items():
                if target in cols_found: continue
                for synonym in synonyms:
                    if synonym in cols_found:
                        rename_dict[synonym] = target
                        break 
            
            if rename_dict:
                df = df.rename(columns=rename_dict)
                print(f"Columnas Re-Ordenadas -> {rename_dict}")

            return df

        except Exception as e:
            msg_error.set(f"Error de Lectura: {str(e)}")
            return None

    @reactive.Effect
    @reactive.event(input.file_upload)
    def load_csv():
        msg_error.set("") 
        file_info = input.file_upload()
        if not file_info: return
        path = file_info[0]["datapath"]
        name = file_info[0]["name"]
        df_loaded = smart_load(path, name)
        if df_loaded is not None:
            val.set(process_dataframe(df_loaded))
        else:
            val.set(None)

    @reactive.Effect
    @reactive.event(input.btn_example)
    def load_example():
        msg_error.set("")
        data = {
            'PhenotypeList': ['Usher Syndrome type 1', 'Benign polymorphism', 'Cardiomyopathy', 'Not specified', 
                              'Long QT Syndrome', 'Silent mutation', 'Ehlers-Danlos', 'Unknown variant'],
            'Type': ['Deletion', 'single nucleotide variant', 'Duplication', 'single nucleotide variant',
                     'Insertion', 'single nucleotide variant', 'Deletion', 'Indel']
        }
        val.set(process_dataframe(pd.DataFrame(data)))

    def process_dataframe(df_input):
        if df_input is None: return None
        
        if 'PhenotypeList' not in df_input.columns:
            df_input['PhenotypeList'] = "not specified"
        df_input['PhenotypeList'] = df_input['PhenotypeList'].fillna('')
        
        try:
            X_text = vectorizer.transform(df_input['PhenotypeList'])
            text_features = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())
            
            if 'Type' not in df_input.columns:
                df_input['Type'] = 'single nucleotide variant'
                
            type_dummies = pd.get_dummies(df_input['Type'], prefix='type')
            
            X_final = pd.concat([text_features, type_dummies], axis=1)
            expected_cols = rf_model.feature_names_in_
            X_final = X_final.reindex(columns=expected_cols, fill_value=0)
            
            probs = rf_model.predict_proba(X_final)[:, 1]
            
            predicciones = []
            for p in probs:
                if p >= 0.70:
                    predicciones.append("[!] PATOGÉNICO")
                elif p <= 0.30:
                    predicciones.append("[OK] BENIGNO")
                else:
                    predicciones.append("[?] INCIERTO")
            
            df_input['PRED'] = predicciones
            df_input['CONF_PCT'] = (probs * 100).round(1)
            
            df_input['Hover_Info'] = df_input['PhenotypeList'].astype(str).str.slice(0, 60) + "..."
            
            return df_input
        except Exception as e:
            msg_error.set(f"Error de Procesamiento: {e}")
            return None

    @output
    @render.text
    def status_text():
        if msg_error.get(): return f"ERROR_LOG: {msg_error.get()}"
        if val.get() is None: return "Esperando carga de datos :p ..."
        return f"Procesamiento Completo // Filas: {len(val.get())}"

    @output
    @render.data_frame
    def prediction_table():
        df = val.get()
        if df is None: return pd.DataFrame()
        desired_cols = ['PRED', 'CONF_PCT', 'Type', 'PhenotypeList', 'GeneSymbol']
        final_cols = [c for c in desired_cols if c in df.columns]
        return df[final_cols]
    
    @render.download(filename="prediction_model_logs.csv")
    def download_pred():
        df = val.get()
        if df is not None:
            yield df.to_csv(index=False)

    @render_widget
    def interactive_plot():
        df = val.get()
        if df is None: return None
        
        color_map = {
            '[!] PATOGÉNICO': "#d6796e",  
            '[?] INCIERTO': "#e3cd78",
            '[OK] BENIGNO': "#7ddba4"      
        }

        orden_eje = ["[!] PATOGÉNICO", "[?] INCIERTO", "[OK] BENIGNO"]

        fig = px.box(
            df, 
            x="CONF_PCT", 
            y="PRED", 
            color="PRED",
            color_discrete_map=color_map,
            category_orders={"PRED": orden_eje},
            orientation="h",
            points="all",
            hover_data={
                "PRED": False,
                "CONF_PCT": True,
                "Type": True,
                "Hover_Info": True 
            },
            title=">> Cluster de Análisis Interactivo"
        )
        
        fig.update_layout(
            font_family="Courier New",
            plot_bgcolor="white",
            xaxis_title="NIVEL DE CONFIANZA (%)",
            yaxis_title=None,
            showlegend=False,
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Courier New"
            )
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eee', range=[-5, 105])
        
        fig.update_traces(
            marker=dict(size=9, opacity=0.7, line=dict(width=1, color='black')),
            hoveron="points" 
        )
        
        return fig

app = App(app_ui, server)
