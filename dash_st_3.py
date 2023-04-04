# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:47:49 2023

@author: kbouh
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd

import json

#import joblib
#from joblib import load
import pickle
#import gzip

import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from lime import lime_tabular


def application(environ, start_response):
  if environ['REQUEST_METHOD'] == 'OPTIONS':
    start_response(
      '200 OK',
      [
        ('Content-Type', 'application/json'),
        ('Access-Control-Allow-Origin', '*'),
        ('Access-Control-Allow-Headers', 'Authorization, Content-Type'),
        ('Access-Control-Allow-Methods', 'POST'),
      ]
    )
    return ''

df = pd.read_csv('data_reduced3.csv')
df = df.drop(['Unnamed: 0'], axis=1)
liste_id = df['SK_ID_CURR'].tolist()
classifier = pickle.load(open("DummyClassifier.pkl", 'rb'))
#classifier = pickle.loads(gzip.open("RFClassifier3.pkl.gz","rb").read())



SK_ID_CURR=st.text_input('Veuillez saisir l\'identifiant d\'un client:', )

body = {
    'SK_ID_CURR':SK_ID_CURR
    }



st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")





tab1, tab2, tab3, tab4  = st.tabs(["Client", "Caractéristiques locales", "Caractéristiques globales", "Comparaison à l'ensemble des clients"])


if st.button("Crédit accordé ou refusé ?"):
    
    
    if int(body.get('SK_ID_CURR')) in liste_id:
        
        with tab1:
           
            col1, col2 = st.columns(2)
       
            with col1:
                st.header("Prédiction : ")
                res1 = requests.post('https://app-loan-fastapi.herokuapp.com/docs#/default/predict_target_predict_post', data = json.dumps(body))
                if (float(res1.json())[0]) <= 0.3:
                    prediction="Crédit refusé"
                    
                   
                elif(float(res1.json())[0]) > 0.3:
                    prediction="Crédit accordé"
                    
                else:
                    "Erreur"
                st.subheader(prediction)
              
            with col2:
                st.header("La jauge de prédiction : ")
                fig = go.Figure(go.Indicator(
                     domain = {'x': [0, 1], 'y': [0, 1]},
                     value = float((res1.json())[0]),
                     mode = "gauge+number",
                     title = {'text': "Score client"},
                     delta = {'reference': 1},
                     gauge = {'axis': {'range': [None, 1]},
                     'steps' : [
                                  {'range': [0, 0.5], 'color': "lightgray"},
                                  {'range': [0.5, 1], 'color': "gray"}],
                              'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 490}}))
                
                
                fig.update_layout(paper_bgcolor='white',
                                 height=250, width=300,
                                 font={'color': '#772b58', 'family': 'Roboto Condensed'},
                                 margin=dict(l=30, r=30, b=5, t=5))
                
                
                st.plotly_chart(fig)
      
        with tab2:
           
           #if st.button("Crédit accordé ou refusé ?"):
           
            st.header("Principales features au niveau local: ")
            
            X = df.drop(['SK_ID_CURR','TARGET'], axis=1)
            y = df['TARGET']
            X_train, X_test, y_train, y_test = \
                            train_test_split(X, 
                                             y,  
                                             test_size = 0.2, 
                                             random_state = 42
                                            )
            
            explainer = lime_tabular.LimeTabularExplainer(X_train.to_numpy(), 
                                                  mode="classification",  
                                                  feature_names= list(X_train.columns))
           
            
            idx = liste_id.index(int(body.get('SK_ID_CURR')))
            
                
            exp = explainer.explain_instance(X_test.iloc[idx], classifier.predict_proba,
                                             num_features=len(list(X_test.columns)))
    
            fig = exp.as_pyplot_figure()
            
            st.pyplot(fig)
           
        with tab3:
           
           #if st.button("Crédit accordé ou refusé ?"):
           
            st.header("Principales features au niveau global: ")
            st.image('feature_importance_RFC.png')
           
      
        with tab4:
            
            C = df.drop(['SK_ID_CURR','TARGET'], axis=1).columns
           
            option = st.selectbox('Choisir une caractéristique',
                (C))
    
            XX = pd.DataFrame([{
                         'score_client': df[option][df[option].index[df['SK_ID_CURR']==int(body.get('SK_ID_CURR'))][0]], 
                         'score_moyen': df[option].mean(), 
                         'score_min': df[option].min(),
                         'score_max': df[option].max()}])
            
          
            data = XX.iloc[0].to_dict()
            
            
            st.bar_chart(data)
                
                
                
    else: 
            {'Client pas identifié'}
    

   
   
   