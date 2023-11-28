import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import pickle
from urllib.request import urlopen
import json
from PIL import Image





st.markdown(
    """
    <style>
    .main{
        background-color:#F5F5F5;
    }
    <style>
    """
)


# Main sections
header = st.container()
dataset= st.container()
features = st.container()
model_training = st.container()

# load our best model
PATH = "C:/Users/DELL/Formation OC/Are you bankable/Datas/"

#Load Dataframe

X_test=pd.read_csv(PATH+'X_test.csv')
y_test=pd.read_csv(PATH+'y_test.csv')
dataframe=pd.read_csv(PATH+'df_test.csv')




@st.cache #mise en cache de la fonction pour exécution unique
def chargement_ligne_data(id, df):
    return df[df['SK_ID_CURR']==int(id)].drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)



@st.cache_data #Cette fonction ne sera exécutée qu'une seule fois
def get_data(filename):
    credit_data = pd.read_csv(filename)
    return credit_data


@st.cache #mise en cache de la fonction pour exécution unique
def chargement_ligne_data(id, df):
    return df[df['SK_ID_CURR']==int(id)].drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)



### Sidebar
st.sidebar.title("Menus")
sidebar_selection = st.sidebar.radio(
    'Select Menu:',
    ['Overview', 'Data Analysis', 'Model & Prediction','Prédire solvabilité client'],
    )


with header:
    st.title("Make sure a client is a sure client!") 
    

    
with dataset:
    st.header("The train dataset is made of a selection of relevant features chosen after EDA")
    credit_data = dataframe
    st.write (credit_data.head())

    
    
with model_training:
    # st.header("Time to train our chosen model!")
    # st.text("You can choose some hyperparameters for the chosen model")
    
    model_selection_column, display_column = st.columns(2)
    test_clients = pd.read_csv(PATH+'df_test.csv')
    liste_id = test_clients['SK_ID_CURR'].tolist()

     # Choose a client

    # chosen_client = str(model_selection_column.selectbox("Please chose your client ID", test_clients['SK_ID_CURR']))
    chosen_client = st.text_input('Veuillez saisir l\'identifiant d\'un client:', )

    st.success("client chosen")
    
    if chosen_client == '':
        st.write('Veuillez recommencer')
        
    elif (int(chosen_client) in liste_id) :
          
        # On peut appeler l'API
          
        API_url = "http://127.0.0.1:5000/credit/" + chosen_client
          
        with st.spinner('Attente du score du client choisi ...'):
            
          json_url = urlopen(API_url)
          
          API_data = json.loads(json_url.read())
          classe_predite = API_data["prediction"]
          if classe_predite == 1:
              resultat = "client dangereux"
          else:
              resultat = "client peu risqué"
          
          proba = 1- API_data["proba"]
          
          #affichage de la prédiction
          prediction = API_data['proba']
          # classe_reelle = dataframe[dataframe['SK_ID_CURR']==int(chosen_client)]['LABELS'].values[0]
          # classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
          chaine = 'Prédiction : **' + resultat +  '** avec **' + str(round(proba*100)) + '%** de risque de défaut '

        st.markdown(chaine)

        st.subheader("Caractéristiques influençant le score")
          
          
#MLFLOW tracking    
# Set the experiment
# Mlflow tracking

    track_with_mlflow = st.checkbox(
        "📈 Track with mlflow? ", help="Mark to track experiment with MLflow"
    )

    # Model training
    start_training = st.button("💪 Start training", help="Train and evaluate ML model")
    if not start_training:
        st.stop()

    if track_with_mlflow:
        mlflow.set_experiment(data_choice)
        mlflow.start_run()
        mlflow.log_param("model", model_choice)
        mlflow.log_param("features", feature_choice)


mlflow.set_experiment("optimized_RF_Classifier")

# Log a metric
accuracy = 0.9
mlflow.log_metric("accuracy", accuracy)

# Log an artifact
model = pickle.dumps(my_model)
mlflow.log_artifact("model", model)

# Display the metrics and artifacts
st.write("Accuracy:", accuracy)
st.write("Model:", model)
    
    