import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import gradio as gr
import numpy as np

# Charger les données
df = pd.read_csv('F1_Data.csv')
print(df.columns.tolist())  # Affiche toutes les colonnes


# Prétraitement des données
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Remplir les NaN dans la colonne 'position' par 0
df = df.assign(position=df['position'].fillna(0))

# Supprimer les lignes avec des NaN dans les colonnes clés
df_clean = df.dropna(subset=['position', 'races_name', 'circuit_name', 'location', 'position_départ', 'Rainfall']).copy()

# Créer les features et le label
features = ['races_name', 'circuit_name', 'location', 'year', 'month', 'day', 'Rainfall', 'position_départ']
X = df_clean[features].copy()
y = df_clean['position'].copy()

# print(f"Dimensions de X : {X.shape}")
# print(f"Dimensions de y : {y.shape}")
# print(f"Nombre de NaN dans y (position): {y.isna().sum()}")

# Convertir les features catégorielles en variables numériques
X = pd.get_dummies(X, columns=['races_name', 'circuit_name', 'location', 'Rainfall'], drop_first=True)

# Diviser les données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle de machine learning (RandomForest pour cet exemple)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer la précision du modèle
from sklearn.metrics import accuracy_score
print(f"Précision : {accuracy_score(y_test, y_pred)}")


def predict_top10(driver_forename, driver_surname, races_name, circuit_name, location, date_str, rainfall, position_depart):
    print(f"Valeurs entrées : {races_name}, {circuit_name}, {location}, '{date_str}', {rainfall}, {position_depart}, {driver_forename}, {driver_surname}")
    try:
        print(f"Date entrée par l'utilisateur : {date_str}")
        # Vérifier si la date est dans le bon format
        if not isinstance(date_str, str) or len(date_str) != 10:
            return "Erreur : Format de date incorrect. Utilisez DD/MM/YYYY."

        # Convertir la date en 'year', 'month', 'day'
        date = pd.to_datetime(date_str, format='%d/%m/%Y', errors='coerce')

        # Vérifier si la conversion a réussi
        if pd.isnull(date):
            return "Erreur : Format de date incorrect. Utilisez DD/MM/YYYY."

        year = date.year
        month = date.month
        day = date.day

        # Filtrer le DataFrame pour l'année spécifiée
        filtered_df = df[df['year'] == year]

        # Préparer les données pour la prédiction
        input_data = pd.DataFrame({
            'races_name': [races_name],
            'circuit_name': [circuit_name],
            'location': [location],
            'year': [year],
            'month': [month],
            'day': [day],
            'Rainfall': [rainfall],
            'position_départ': [position_depart]
        })

        # Appliquer les mêmes transformations que lors de l'entraînement
        input_data = pd.get_dummies(input_data, columns=['races_name', 'circuit_name', 'location', 'Rainfall'],
                                    drop_first=True)

        # Aligner les colonnes avec celles du modèle
        input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

        # Prédire les positions
        predictions = model.predict(input_data)

        # Regrouper par le nom du conducteur et garder la première ID


        results = []
        seen_ids = set()  # Pour suivre les identifiants déjà utilisés

        # Obtenir les positions uniques
        unique_positions = pd.Series(predictions).unique()

        for pos in unique_positions:
            matching_drivers = filtered_df[filtered_df['position'] == pos]

            for driver_info in matching_drivers.itertuples():
                driver_id = getattr(driver_info, 'driverId')
                if driver_id not in seen_ids:  # Vérifier l'identifiant
                    results.append(
                        (driver_info.forename, driver_info.surname, driver_info.constructor_name))  # Trois valeurs
                    seen_ids.add(driver_id)  # Ajouter l'identifiant aux vus
                    if len(results) >= 10:  # Limiter à 10 résultats
                        break
            if len(results) >= 10:
                break

        # Formatage des résultats
        output = []
        for idx, (forename, surname, constructor_name) in enumerate(results):  # Trois valeurs
            position_suffix = "th" if 4 <= idx + 1 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get((idx + 1) % 10, "th")
            print(results)
            output.append(f"{idx + 1}{position_suffix} - {forename} {surname} ({constructor_name})")

        return "\n".join(output) if output else "Aucun pilote trouvé."
    except Exception as e:
        return f"Erreur : {str(e)}"


import gradio as gr

# Interface Gradio
with gr.Blocks(css="""
    body {background-color: #f0f0f0; font-family: 'Arial', sans-serif;}
    .gradio-container {background-color: #f5f5f5; padding: 20px;}
    h1 {color: #333; text-align: center; font-family: 'Helvetica', sans-serif;}
    p {font-size: 16px; color: #666;}
""") as interface:
    # Bannière d'image en haut de la page
    gr.Image("https://via.placeholder.com/800x200", label="Bannière", show_label=False)

    # Titre
    gr.Markdown("<h1>F1 Classement Simulator</h1>")

    # Description
    gr.Markdown("<p>Enter informations and see if your favorite pilot is in the classement!</p>")

    # Formulaire d'entrée
    with gr.Row():
        driver_surname = gr.Textbox(label="Driver Last name")
        driver_forename = gr.Textbox(label="Driver First name")
        races_name = gr.Textbox(label="Race Name", placeholder="")
        circuit_name = gr.Textbox(label="Circuit Name", placeholder="")
        location = gr.Textbox(label="Localisation", placeholder="")
        date = gr.Textbox(label="Date (DD/MM/YYYY)", placeholder="Ex: 25/07/2023")
        rainfall = gr.Radio(label="Rain ?", choices=["True", "False"])
        position_depart = gr.Slider(minimum=1, maximum=22, step=1, label="Grid Position")

    # Sortie
    output = gr.Textbox(label="The Champions")

    # Bouton de soumission
    submit_btn = gr.Button("Predict")

    # Associer la fonction à l'interface
    submit_btn.click(predict_top10,
                     [driver_surname, driver_forename, races_name, circuit_name, location, date, rainfall, position_depart],
                     output)

interface.launch()

