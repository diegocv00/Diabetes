import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

def main():
    # Cargar el dataset
    try:
        df = pd.read_csv("diabetes_dataset.csv")
        print("Dataset cargado exitosamente.")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'diabetes_dataset.csv'. Asegúrate de que esté en el directorio correcto.")
        return

    # Limpieza y preparación de datos
    if 'income_level' in df.columns:
        df["income_level"] = df["income_level"].replace({
            "Low": "Bajo",
            "Middle": "Medio",
            "High": "Alto",
            "Lower-Middle": "Medio-Bajo",
            "Upper-Middle": "Medio-Alto"
        })
    
    if 'smoking_status' in df.columns:
        df["smoking_status"] = df["smoking_status"].replace({
            "Never": "Nunca",
            "Former": "Anteriormente",
            "Current": "Actualmente"
        })

    # Funciones de clasificación
    def clasificar_imc(x):
        if x < 18.5:
            return "Bajo peso"
        elif 18.5 <= x < 24.9:
            return "Normal"
        elif 25 <= x < 29.9:
            return "Sobrepeso"
        else:
            return "Obesidad"

    if 'bmi' in df.columns:
        df['imc_category'] = df['bmi'].apply(clasificar_imc)

    def clasificar_hr(x):
        if x < 60:
            return "Bradicardia"
        elif 60 <= x <= 100:
            return "Normal"
        else:
            return "Taquicardia"

    if 'heart_rate' in df.columns:
        df['heart_rate_category'] = df['heart_rate'].apply(clasificar_hr)

    # Clasificación de Género
    def clasificar_genero(x):
        if x == "Male":
            return "Masculino"
        elif x == "Female":
            return "Femenino"
        else:
            return "Otro"
    
    if 'gender' in df.columns:
        df['gender'] = df['gender'].apply(clasificar_genero)
    
    def clasificar_glucosa_ayunas(x):
        if x < 100:
            return "Normal"
        elif 100 <= x <= 125:
            return "Prediabetes"
        else:
            return "Diabetes"

    if 'fasting_glucose' in df.columns:
        df['glucose_fasting_category'] = df['fasting_glucose'].apply(clasificar_glucosa_ayunas)

    def clasificar_glucosa_postprandial(x):
        if x < 140:
            return "Normal"
        elif 140 <= x <= 199:
            return "Prediabetes"
        else:
            return "Diabetes"

    if 'glucose_postprandial' in df.columns:
        df['glucose_postprandial_category'] = df['glucose_postprandial'].apply(clasificar_glucosa_postprandial)
    
    def clasificar_hba1c(x):
        if x < 5.7:
            return "Normal"
        elif 5.7 <= x <= 6.4:
            return "Prediabetes"
        else:
            return "Diabetes"
    
    if 'hba1c' in df.columns:
        df['hba1c_category'] = df['hba1c'].apply(clasificar_hba1c)
    
    # Separar dataframes
    df_poseen_diabetes = df[df["diagnosed_diabetes"] == 1]
    df_no_poseen_diabetes = df[df["diagnosed_diabetes"] == 0]

    # Nombre del archivo PDF de salida
    output_pdf = "diabetes_analisis_reporte.pdf"
    print(f"Generando reporte PDF: {output_pdf}...")

    # Función auxiliar para añadir texto de análisis a la figura
    def add_analysis_text(fig, text):
        # Ajustar el layout para dejar más espacio abajo y evitar solapamientos
        plt.subplots_adjust(bottom=0.3)
        # Envolver el texto para que quepa en la figura
        wrapped_text = "\n".join(textwrap.wrap(text, width=90))
        
        # Estilo del cuadro de texto
        bbox_props = dict(boxstyle="round,pad=1", fc="#f4f6f7", ec="#b0bec5", alpha=0.9, linewidth=1.5)
        
        fig.text(0.5, 0.02, 
                 f"{wrapped_text}", 
                 ha='center', va='bottom', 
                 fontsize=11, fontfamily='serif', color='#37474f',
                 bbox=bbox_props)

    with PdfPages(output_pdf) as pdf:
        
        # 1. Género
        fig = plt.figure(figsize=(10, 8))
        sns.countplot(data=df, x="gender", hue="diagnosed_diabetes")
        plt.xlabel("Género")
        plt.ylabel("Cantidad de pacientes")
        plt.title("Distribución de diagnóstico de diabetes por género")
        plt.legend(title="Diagnóstico de diabetes", labels=["No", "Sí"])
        analysis_text = "No se observa una gran diferencia entre si tiene diabetes y el género de las personas"
        add_analysis_text(fig, analysis_text)
        pdf.savefig(fig)
        plt.close()

        # 2. Nivel de Ingreso
        fig = plt.figure(figsize=(10, 8))
        sns.countplot(data=df, x="income_level", hue="diagnosed_diabetes")
        plt.xlabel("Nivel de ingreso")
        plt.ylabel("Cantidad de pacientes")
        plt.title("Distribución de diagnóstico de diabetes por nivel de ingreso")
        plt.legend(title="Diagnóstico de diabetes", labels=["No", "Sí"])
        analysis_text = "Las personas ubicadas en el nivel de ingreso medio son las que poseen mas diabetes,esto quizas es debido a que tienen mayor poder adquisitivo que los de nivel bajo pero no tienen el mismo acceso a la calidad de alimento y mejores servicios de salud de los de nivel alto"
        add_analysis_text(fig, analysis_text)
        pdf.savefig(fig)
        plt.close()

        # 3. Fumar
        fig = plt.figure(figsize=(10, 8))
        sns.countplot(data=df, x="smoking_status", hue="diagnosed_diabetes")
        plt.xlabel("Estado de fumador")
        plt.ylabel("Cantidad de pacientes")
        plt.title("Distribución de diagnóstico de diabetes por estado de fumador")
        plt.legend(title="Diagnóstico de diabetes", labels=["No", "Sí"])
        analysis_text = "Aunque la gráfica muestra algo contraintuitivo esto se puede deber a que la proporcion de no fumadores es mucho mayor respecto a los otros dos estados, tambien a un efecto de superviviencia, muchos fumadores no llegan a ser diagnosticados con diabetes antes de su muerte. Y finalmente una compensacion de ansiedad con los no fumadores se lleva a cabo con el alimento lo cual ayuda al diagnostico de diabetes."
        add_analysis_text(fig, analysis_text)
        pdf.savefig(fig)
        plt.close()

        # 4. Consumo de Alcohol
        porcentajes_por_consumo_de_alcohol = df.groupby("alcohol_consumption_per_week")["diagnosed_diabetes"].value_counts(normalize=True).unstack().fillna(0) * 100 
        fig = plt.figure(figsize=(12, 6))
        plt.bar(porcentajes_por_consumo_de_alcohol.index, porcentajes_por_consumo_de_alcohol[1], label='Con diabetes', color='salmon')
        plt.bar(porcentajes_por_consumo_de_alcohol.index, porcentajes_por_consumo_de_alcohol[0], bottom=porcentajes_por_consumo_de_alcohol[1], label='Sin diabetes', color='lightblue')
        plt.xlabel("Consumo de alcohol por semana")
        plt.ylabel("Porcentaje de pacientes")
        plt.title("Porcentaje de diagnóstico de diabetes según consumo de alcohol por semana")
        plt.legend(labels=["Sin diabetes", "Con diabetes"])
        analysis_text = "Con un consumo relativamente moderado (de 0-7 veces por semana) el porcentaje de no poseer diabetes es alto(alrededor del 60%), mientras cuando el consumo ya supera 1 vez diaria este porcentaje cae, aunque es muy poco."
        add_analysis_text(fig, analysis_text)
        pdf.savefig(fig)
        plt.close()

        # 5. Actividad Física
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        if not df_no_poseen_diabetes.empty:
            ax1.hist(df_no_poseen_diabetes['physical_activity_minutes_per_week'], bins=50, color='#8CFFDF')
            ax1.set_title("Actividad física (sin diabetes)")
            ax1.set_xlabel("Minutos de actividad física por semana")

        if not df_poseen_diabetes.empty:
            ax2.hist(df_poseen_diabetes['physical_activity_minutes_per_week'], bins=50, color='#FFDD8C')
            ax2.set_title("Actividad física (con diabetes)")
            ax2.set_xlabel("Minutos de actividad física por semana")

        analysis_text = "Las personas con diabetes tienden a realizar menos actividad física, concentrándose en el rango de 0 a 100 minutos semanales."
        add_analysis_text(fig, analysis_text)
        pdf.savefig(fig)
        plt.close()

        # 6. Horas de Sueño
        fig = plt.figure(figsize=(10, 8))
        plt.hist(df_poseen_diabetes['sleep_hours_per_day'], bins=40, alpha=0.7, label='Con Diabetes', color='purple')
        plt.xlabel("Horas de sueño por día")
        plt.ylabel("Cantidad de pacientes")
        plt.title("Distribución de horas de sueño por día (pacientes con diabetes)")
        plt.legend()
        analysis_text = "La mayoría de las personas duerme entre 4 y 9 horas, con un pico en 8 horas. No se observa una relación directa fuerte con la diabetes en este gráfico."
        add_analysis_text(fig, analysis_text)
        pdf.savefig(fig)
        plt.close()

        # 7. IMC
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        has_data = False
        if not df_no_poseen_diabetes.empty:
            counts_no = df_no_poseen_diabetes['imc_category'].value_counts()
            ax1.pie(counts_no, labels=counts_no.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            ax1.set_title("IMC en pacientes sin diabetes")
            has_data = True
        
        if not df_poseen_diabetes.empty:
            counts_si = df_poseen_diabetes['imc_category'].value_counts()
            ax2.pie(counts_si, labels=counts_si.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            ax2.set_title("IMC en pacientes con diabetes")
            has_data = True
            
        if has_data:
            analysis_text = "Se observa una mayor proporción de obesidad en el grupo de pacientes con diabetes en comparación con los que no la tienen."
            add_analysis_text(fig, analysis_text)
            pdf.savefig(fig)
        plt.close()

        # 8. Antecedentes Familiares
        if "family_history_diabetes" in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
            
            df_antecedentes = df.groupby("family_history_diabetes")["diagnosed_diabetes"].value_counts(normalize=True).unstack().fillna(0).reset_index()
            df_con = df_antecedentes[df_antecedentes["family_history_diabetes"] == 1]
            df_sin = df_antecedentes[df_antecedentes["family_history_diabetes"] == 0]

            has_fam_data = False
            if not df_sin.empty and df_sin.shape[1] >= 3:
                ax1.pie(df_sin.iloc[0, 1:].values, labels=["Sin diabetes", "Con diabetes"], autopct='%1.1f%%', colors=['lightblue', 'salmon'])
                ax1.set_title("Diagnóstico: sin antecedentes familiares")
                has_fam_data = True

            if not df_con.empty and df_con.shape[1] >= 3:
                ax2.pie(df_con.iloc[0, 1:].values, labels=["Sin diabetes", "Con diabetes"], autopct='%1.1f%%', colors=['lightblue', 'salmon'])
                ax2.set_title("Diagnóstico: con antecedentes familiares")
                has_fam_data = True
            
            if has_fam_data:
                analysis_text = "Como es esperable, las personas con antecesores con diabetes son mas propensas a poseer diabetes. Mientras que la distribucion de persona sin antecesores con diabetes es casi la misma."
                add_analysis_text(fig, analysis_text)
                pdf.savefig(fig)
            plt.close()

        # 9. Frecuencia Cardíaca
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        if not df_no_poseen_diabetes.empty:
            counts_no_hr = df_no_poseen_diabetes['heart_rate_category'].value_counts()
            bar1 = ax1.bar(counts_no_hr.index, counts_no_hr, color=sns.color_palette("pastel"))
            ax1.set_title("Frecuencia cardíaca (con diabetes)")
            ax1.set_ylabel("Cantidad")
            ax1.bar_label(bar1, padding=3)

        if not df_poseen_diabetes.empty:
            counts_si_hr = df_poseen_diabetes['heart_rate_category'].value_counts()
            bar2 = ax2.bar(counts_si_hr.index, counts_si_hr, color=sns.color_palette("pastel"))
            ax2.set_title("Frecuencia cardíaca (con diabetes)")
            ax2.set_ylabel("Cantidad")
            ax2.bar_label(bar2, padding=3)
            
        analysis_text = "No se observa una gran relación entre una frecuencia cardíaca alta y el poseer o no diabetes. Además la mayoria de pacientes tiene una frecuencia cardiaca normal, entre 60-100 latidos."
        add_analysis_text(fig, analysis_text)
        pdf.savefig(fig)
        plt.close()

        # 10. Colesterol HDL
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        if not df_no_poseen_diabetes.empty:
            ax1.hist(df_no_poseen_diabetes['hdl_cholesterol'], bins=50, color='#8CFFDF')
            ax1.set_title("HDL (sin diabetes)")
            ax1.set_xlabel("Colesterol HDL")

        if not df_poseen_diabetes.empty:
            ax2.hist(df_poseen_diabetes['hdl_cholesterol'], bins=50, color='#FFDD8C')
            ax2.set_title("HDL (con diabetes)")
            ax2.set_xlabel("Colesterol HDL")
            
        analysis_text = "Los pacientes con diabetes muestran una tendencia a tener niveles más bajos de HDL, lo que reduce su protección contra enfermedades del corazón."
        add_analysis_text(fig, analysis_text)
        pdf.savefig(fig)
        plt.close()

        # 11. Colesterol LDL
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        if not df_no_poseen_diabetes.empty:
            ax1.hist(df_no_poseen_diabetes['ldl_cholesterol'], bins=50, color='#8CFFDF')
            ax1.set_title("LDL (sin diabetes)")
            ax1.set_xlabel("Colesterol LDL")

        if not df_poseen_diabetes.empty:
            ax2.hist(df_poseen_diabetes['ldl_cholesterol'], bins=50, color='#FFDD8C')
            ax2.set_title("LDL (con diabetes)")
            ax2.set_xlabel("Colesterol LDL")
            
        analysis_text = "Aunque las gráficas se ven parecidas, en pacientes diabéticos el problema no es solo la cantidad de LDL, sino su calidad. Además, en el gráfico con diabetes se observa una 'cola' ligeramente más larga hacia la derecha."
        add_analysis_text(fig, analysis_text)
        pdf.savefig(fig)
        plt.close()

        # 12. Triglicéridos
        promedio_tg_no_diabetes = df_no_poseen_diabetes['triglycerides'].mean()
        promedio_tg_con_diabetes = df_poseen_diabetes['triglycerides'].mean()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        if not df_no_poseen_diabetes.empty and 'triglycerides' in df.columns:
            ax1.hist(df_no_poseen_diabetes['triglycerides'], bins=50, color='#8CFFDF')
            ax1.set_xlabel("Triglicéridos (mg/dL)")
            ax1.set_ylabel("Cantidad de pacientes")
            ax1.set_title("Distribución de triglicéridos en pacientes sin diabetes")
            ax1.grid(axis='y', alpha=0.3)
            ax1.axvline(promedio_tg_no_diabetes, color='black', linestyle='dashed', linewidth=2, label=f'Promedio: {promedio_tg_no_diabetes:.2f}')
            ax1.text(promedio_tg_no_diabetes + 5, ax1.get_ylim()[1] * 0.9, f'Media: {promedio_tg_no_diabetes:.2f}', color='black', fontweight='bold')  
    
        if not df_poseen_diabetes.empty and 'triglycerides' in df.columns:
            ax2.hist(df_poseen_diabetes['triglycerides'], bins=50, color='#FFDD8C')
            ax2.set_title("Distribución de triglicéridos en pacientes con diabetes")
            ax2.set_xlabel("Triglicéridos (mg/dL)")
            ax2.set_ylabel("Cantidad de pacientes")
            ax2.grid(axis='y', alpha=0.3)
            ax2.axvline(promedio_tg_con_diabetes, color='black', linestyle='dashed', linewidth=2, label=f'Promedio: {promedio_tg_con_diabetes:.2f}')
            ax2.text(promedio_tg_con_diabetes + 5, ax2.get_ylim()[1] * 0.9, f'Media: {promedio_tg_con_diabetes:.2f}', color='black', fontweight='bold')
            
        analysis_text = "Los pacientes con diabetes presentan triglicéridos más elevados en promedio. Esto es un factor de riesgo cardiovascular importante asociado con la diabetes"
        add_analysis_text(fig, analysis_text)
        pdf.savefig(fig)
        plt.close()

        # 13. Glucosa en Ayunas
        if 'fasting_glucose' in df.columns:
            df_poseen_diabetes = df[df["diagnosed_diabetes"] == 1]
            df_no_poseen_diabetes = df[df["diagnosed_diabetes"] == 0]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

            ax1.pie(df_no_poseen_diabetes['glucose_fasting_category'].value_counts(), labels=df_no_poseen_diabetes['glucose_fasting_category'].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            ax1.set_title("Distribución de glucosa en ayunas en pacientes sin diabetes")
            counts_no_diabetes = df_no_poseen_diabetes['glucose_fasting_category'].value_counts().to_dict()


            ax2.pie(df_poseen_diabetes['glucose_fasting_category'].value_counts(), labels=df_poseen_diabetes['glucose_fasting_category'].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            ax2.set_title("Distribución de glucosa en ayunas en pacientes con diabetes")
            counts_with_diabetes = df_poseen_diabetes['glucose_fasting_category'].value_counts().to_dict()

            analysis_text = "La glucosa en ayunas es un indicador clave de diabetes. Se observa que existe gran cantidad de personas con riesgo de diabetes debido a la glucosa en ayunas"
            add_analysis_text(fig, analysis_text)
            pdf.savefig(fig)
            plt.close()

        # 14. Glucosa Postprandial
        if 'glucose_postprandial' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
            
            counts_no = df_no_poseen_diabetes['glucose_postprandial_category'].value_counts()
            bar1 = ax1.bar(counts_no.index, counts_no, color=sns.color_palette("pastel"))
            ax1.set_title("Distribución de la glucosa postprandial en pacientes sin diabetes")
            ax1.set_xlabel("Glucosa postprandial (mg/dL)")
            ax1.set_ylabel("Cantidad de pacientes")
            ax1.bar_label(bar1, padding=3)


            counts_si = df_poseen_diabetes['glucose_postprandial_category'].value_counts()
            bar2 = ax2.bar(counts_si.index, counts_si, color=sns.color_palette("pastel"))
            ax2.set_title("Distribución de la glucosa postprandial en pacientes con diabetes")
            ax2.set_xlabel("Glucosa postprandial (mg/dL)")
            ax2.set_ylabel("Cantidad de pacientes")
            ax2.bar_label(bar2, padding=3)
            
            analysis_text = "La glucosa postprandial (después de comer) es otro indicador importante. Los pacientes diabéticos muestran un control glucémico deficiente después de las comidas, con valores mucho más elevados que los no diabéticos"
            add_analysis_text(fig, analysis_text)
            pdf.savefig(fig)
            plt.close()

        # 15. Nivel de Insulina
        if 'insulin_level' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
            
            if not df_no_poseen_diabetes.empty:
                ax1.hist(df_no_poseen_diabetes['insulin_level'], bins=50, color='#8CFFDF')
                ax1.set_title("Nivel de Insulina (sin diabetes)")
                ax1.set_xlabel("uIU/mL")

            if not df_poseen_diabetes.empty:
                ax2.hist(df_poseen_diabetes['insulin_level'], bins=50, color='#FFDD8C')
                ax2.set_title("Nivel de Insulina (con diabetes)")
                ax2.set_xlabel("uIU/mL")
            
            analysis_text ="El nivel de insulina es un indicador importante de la resistencia a la insulina. Interesantemente, algunos pacientes con diabetes tipo 2 pueden tener niveles de insulina más altos debido a la resistencia, mientras que los tipo 1 pueden tener niveles bajos"
            add_analysis_text(fig, analysis_text)
            pdf.savefig(fig)
            plt.close()

        # 16. HbA1c
        if 'hba1c' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
            
            ax1.pie(df_no_poseen_diabetes['hba1c_category'].value_counts(), labels=df_no_poseen_diabetes['hba1c_category'].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            ax1.set_title("Distribución de nivel de HbA1c en pacientes sin diabetes")
            counts_no_diabetes = df_no_poseen_diabetes['hba1c_category'].value_counts().to_dict()


            ax2.pie(df_poseen_diabetes['hba1c_category'].value_counts(), labels=df_poseen_diabetes['hba1c_category'].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            ax2.set_title("Distribución de nivel de HbA1c en pacientes con diabetes")
            counts_with_diabetes = df_poseen_diabetes['hba1c_category'].value_counts().to_dict()
            
            analysis_text = "El HbA1c es uno de los mejores indicadores del control glucémico a largo plazo. Es claramente visible que los pacientes con diabetes tienen valores de HbA1c significativamente más altos, reflejando un control glucémico deficiente. El valor de 6.5% es el umbral diagnóstico para diabetes"
            add_analysis_text(fig, analysis_text)
            pdf.savefig(fig)
            plt.close()

        #16. Hipertension
        
        df_antecedentes_familiares_2 = df.groupby("hypertension_history")["diagnosed_diabetes"].value_counts(normalize=True).unstack().fillna(0).reset_index()
        df_poseen_diabetes_antecedentes_familiares_2 = df_antecedentes_familiares_2.loc[df_antecedentes_familiares_2["hypertension_history"] == 1]
        df_no_poseen_diabetes_antecedentes_familiares_2 = df_antecedentes_familiares_2.loc[df_antecedentes_familiares_2["hypertension_history"] == 0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

        ax1.pie(df_no_poseen_diabetes_antecedentes_familiares_2[[1, 0]].values[0], labels=["Con diabetes", "Sin diabetes"], autopct='%1.1f%%', colors=['#78F4FA', '#FA787E'])
        ax1.set_title("Proporción de diagnóstico de diabetes en pacientes sin hipertensión")

        ax2.pie(df_poseen_diabetes_antecedentes_familiares_2[[1, 0]].values[0], labels=["Con diabetes", "Sin diabetes"], autopct='%1.1f%%', colors=['#78F4FA', '#FA787E'])
        ax2.set_title("Proporción de diagnóstico de diabetes en pacientes con hipertensión")

        analysis_text = "No se observa una gran diferencia entre los diabeticos y no diabeticos cuando sufren de hipertensión."
        add_analysis_text(fig, analysis_text)
        pdf.savefig(fig)
        plt.close()

    print(f"Reporte generado exitosamente: {output_pdf}")

if __name__ == "__main__":
    main()
