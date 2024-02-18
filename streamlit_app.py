##########################################################################################################
########             LIBRAIRIES                                                                   ########
##########################################################################################################
import function as fc

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf 
import seaborn as sns
import pickle
from datetime import datetime, timedelta
import plotly.express as px
from PIL import Image
import plotly.graph_objs as go

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="ML & trading", page_icon="👋")

custom_css = """
<style>
    .title {
        color: #ff6347;
        font-size: 80px; 
        text-align: center; 
    }
    .css-1wlt93b {
        padding-top: 0rem;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)
st.markdown('<div class="title">Machine learning & stratégie de trading</div>', unsafe_allow_html=True)

st.balloons()

def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://w.wallhaven.cc/full/9d/wallhaven-9dmeyw.jpg");
             background-size: cover;
             background-attachment: fixed;
             background-repeat: no-repeat;
             background-position: center center;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

add_bg_from_url()


##########################################################################################################
########             PARAMETRES DU DASHBOARD                                                      ########
##########################################################################################################




linkMA = 'https://www.investopedia.com/terms/m/movingaverage.asp'
linkSO = 'https://www.investopedia.com/terms/s/stochasticoscillator.asp'
linkRSI = 'https://www.investopedia.com/terms/r/rsi.asp'
linkROC = 'https://www.investopedia.com/terms/r/rateofchange.asp'
linkMOM = 'https://www.investopedia.com/investing/momentum-and-relative-strength-index/'
linkbook = "https://www.oreilly.com/library/view/machine-learning-and/9781492073048/"
linkgit = "https://github.com/tatsath/fin-ml"
linkNeil = "https://neil-portfolio.netlify.app/"

with st.sidebar:
 

    now = datetime.now()+ timedelta(hours=1)
    time = now.strftime("%H:%M:%S")
    
    st.write("Heure à Paris : ",time)
    
    st.write("####  Paramètres des variables (features) :")

    j = st.slider('Combien de retard (lag) ?', min_value=1, max_value=5, step=1, value =3)

    optionsEMASTORSI = st.multiselect("Délai sur l'indicateur EMA, Stochastic et RSI", [10, 30, 200])
    optionsROCMOM = st.multiselect("Délai sur l'indicateur ROC et MOM" , [10, 30, 50])

    st.write("####  Paramètres avant modélisation :")

    size = st.slider("Taille de l'échantillon de test :", min_value=0.1, max_value=0.3, step=0.05, value =0.25)

    num_folds = st.radio("Sélectionner le nombre de kfold pour la Cross-Validation",key="visibility", options=[5, 10])
    
    st.write("####  Paramètre de prédiction :")
    
    forecast_out = st.slider('Nombre de prédictions (intervalle de 2min)', min_value=10, max_value=30, step=5)

    st.write("####  Lien vers la documation des indicateurs techniques :")

    st.markdown(f'''
<a href={linkMA}><button style="background-color:Linen;">Moving Average </button></a>
''',
                unsafe_allow_html=True)
    st.markdown(f'''
<a href={linkSO}><button style="background-color:Linen;">Oscillateur stochastique </button></a>
''',
                unsafe_allow_html=True)
    st.markdown(f'''
<a href={linkRSI}><button style="background-color:Linen;">RSI</button></a>
''',
                unsafe_allow_html=True)
    st.markdown(f'''
<a href={linkROC}><button style="background-color:Linen;">ROC </button></a>
''',
                unsafe_allow_html=True)
    st.markdown(f'''
<a href={linkMOM}><button style="background-color:Linen;">Momentum </button></a>
''',
                unsafe_allow_html=True)

    st.write("####  Lien vers la documation du code qui a inspiré à la création de ce dashboard :")

    st.markdown(f'''
        <a href={linkbook}><button style="background-color:Linen;">O'Reilly : ML & DS Blueprints for Finance</button></a>
        ''',
                unsafe_allow_html=True)

    st.markdown(f'''
        <a href={linkgit}><button style="background-color:Linen;">Lien vers le Github du livre </button></a>
        ''',
                unsafe_allow_html=True)

from PIL import Image
import streamlit as st

image = Image.open("logo.png")

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image(image)

custom_css = """
<style>
.custom-text {
    color: #FF6347; /* 番茄色 */
    font-size: 24px; /* 字体大小 */
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

st.markdown('<p class="custom-text">Entrez le ticker comme ceci : AAPL, TSLA, AMZN, ... puis appuyez sur la fonction Création de notre dataframe</p>', unsafe_allow_html=True)
ticker = st.text_input(' ', "BTC-USD", placeholder="Entrez votre ticker ... ")


# ##########################################################################################################
# ########             APPEL DES FONCTIONS                                                          ########
# ##########################################################################################################

if len(ticker) != 0:  # si l'utilisateur entre une série de ticker alors l'interface du dashboard se complète
    
    if "button" not in st.session_state:
        st.session_state.button = False

    if (st.button('Création de notre dataframe') or st.session_state.button):

        data, cols = fc.getlastdata(ticker, j, optionsEMASTORSI, optionsROCMOM)
        data = pd.DataFrame(data)
        #st.dataframe(data['direction'].value_counts())
        data = data[203:-1]
        st.write("##### Nos données (après traitement des outliers et avant normalisation) :")
        st.dataframe(data)
        st.write('##### Matrice de corrélation des actifs du portefeuille : ')
        fig = go.Figure(data=go.Heatmap(z=data.corr(),x=data.columns,y=data.columns,hoverongaps = False, texttemplate="%{z}"))
        st.plotly_chart(fig)
        scoring = 'accuracy'
        st.write("##### Moyenne de nos scores de précision (std à droite) avec la cross validation :")
        resultsTrain, resultsTest, names, X_train, Y_train, num_folds,scoring, X_test, Y_test = fc.MLfit(data,
                                                                                     size, num_folds)  # Appel de notre fonction
        resultsTrain = pd.DataFrame(resultsTrain).T
        resultsTrain.columns = names
        resultsTest = pd.DataFrame(resultsTest).T
        resultsTest.columns = names
        resultsTrain["type"] = "Train"
        resultsTest["type"] = "Test"
        results = pd.concat([resultsTrain, resultsTest], ignore_index=True)

        fig4 = px.box(results, color="type")

        st.plotly_chart(fig4)

        st.write("#### Grid Search sur les meilleurs modèles :")

        best_paramRF_criterion, best_paramRF_n_estimators, best_paramRF_max_depth, modelRF = fc.GridSearchForRF(X_train, Y_train, X_test, Y_test, num_folds)

        best_C, best_penalty, modellr = fc.GridSearchForLR(X_train, Y_train, X_test, Y_test)

        best_paramGBM_max_depth, best_paramGBM_n_estimators, modelGBM = fc.GridSearchForGBM(X_train, Y_train,
                                                                                             num_folds, X_test,Y_test)
        
        
        # st.session_state["data"] = data # variable que l'on va utiliser en seconde partie
        # st.session_state["size"] = size # variable que l'on va utiliser en seconde partie
        # st.session_state["cols"] = cols # variable que l'on va utiliser en seconde partie

        # st.session_state["best_paramRF_criterion"] = best_paramRF_criterion
        # st.session_state["best_paramRF_n_estimators"] = best_paramRF_n_estimators
        # st.session_state["best_paramRF_max_depth"]  =best_paramRF_max_depth

        # st.session_state["best_paramGBM_max_depth"] = best_paramGBM_max_depth
        # st.session_state["best_paramGBM_n_estimators"] = best_paramGBM_n_estimators

       #  st.session_state["best_C"] = best_C
       #  st.session_state["best_penalty"] = best_penalty

       # ##########################################################################################################
       # ########             Partie prédiction                                                            ########
   #       st.download_button("Téléchargement du modèle Logistic Regression en format pickle", data=pickle.dumps(modellr), file_name="lr.pkl")

    #    if option == agree2:

    #        uploaded_file = st.file_uploader("Insérer un fichier pickle (pkl) du dernier modèle utilisé (il doit comporter le même nombre de variable que le modèle précédent)")

    #        if uploaded_file is not None:
    #            model = pickle.loads(uploaded_file.read())

    #            st.write("Votre fichier", model)
            # prediction1 = mlAlgoPred(st.session_state["data"], forecast_out, model, st.session_state["size"])
    #            prediction1 = fc.mlAlgoPred(st.session_state["data"], forecast_out, model, st.session_state["size"] , st.session_state["best_paramRF_criterion"] ,st.session_state["best_paramRF_n_estimators"], st.session_state["best_paramRF_max_depth"], st.session_state["best_paramGBM_max_depth"],st.session_state["best_paramGBM_n_estimators"], st.session_state["best_C"], st.session_state["best_penalty"], True)
    #            somme1, conscensus, conscensus2 = fc.OperationOnDF(st.session_state["data"], prediction1, forecast_out, True)

    #            col1, col2 = st.columns(2)

    #            with col1:
    #                st.write("##### Nos données de base :")
    #                st.write(st.session_state["data"])

    #            with col2:
    #                st.write("##### Notre prédiction :")
    #                st.write(conscensus.tail(forecast_out + 1))

    #           st.write("##### Notre prédiction sur les {} prochaines minutes".format(2* forecast_out))
    #           fig = go.Figure()
    #           fig = px.line(conscensus, x="Date", y="direction", text="direction", markers=True)
    #            fig.update_traces(textposition="bottom right")
    #           st.plotly_chart(fig)

    
else:
    st.write("Pas de ticker")
