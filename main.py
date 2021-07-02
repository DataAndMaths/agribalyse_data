# -*- coding: utf-8 -*-

##############################################################################
##############################################################################
#                                                                            #
#                         Importation des librairies                         #
#                                                                            #
##############################################################################
##############################################################################
import streamlit as st

#-----------------------------------#
# general
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------#
# plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, plot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
## fixer le theme
import plotly.io as pio
pio.templates.default = 'ggplot2'


#-----------------------------------#
# coefficient d'asymétrie
from scipy.stats import skew

# pour afficher les informations générales sur le dataset
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report

#-----------------------------------#
# scikit-learn
## train set / test set
from sklearn.model_selection import train_test_split
## encoder
import category_encoders as ce
## preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
## modèles
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# metrics
from sklearn.metrics import *
# model selection 
from sklearn.model_selection import learning_curve, validation_curve
# feature selection
from sklearn.feature_selection import VarianceThreshold, SelectKBest
#------------------#
# Réduction de dimension
from sklearn.decomposition import PCA


##############################################################################
##############################################################################
#                                                                            #
#                         Définition des fonctions                           #
#                                                                            #
##############################################################################
##############################################################################

##############################################################################
# Fonction principale : 
# permet d'afficher les différentes pages
##############################################################################
def main():
    PAGES = {
        "Accueil": page1,
        "Exploration des données": page2,
        "Exploration des données : \n Réduction de dimension" : page2_1,
        "Prédiciton du DQR : Premiers modèles": page3,
        "Prédiciton du DQR : Amélioration des modèles" : page4
        #"Clustering" : page10
        #"Références" : page50
    }

    st.sidebar.title('Navigation 🧭')
    page = st.sidebar.radio("", list(PAGES.keys()))
    PAGES[page]()
    
    
        
##############################################################################
# Fonctions correspondant aux différentes pages
##############################################################################


#==============================   Page 1  ===================================#
#===============================  Accueil  ==================================#
def page1():
    st.title('Agribalyse-Synthèse Application')
    st.write("##")
    st.write("""
             Bienvenue !  
             
             
             """)
             
    st.markdown(":tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:"+
                " :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:"+
                " :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:"+
                " :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:"+
                " :tractor:  :tractor:  :tractor:")         
    
    st.markdown(":apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:"+
                ":apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:"+
                ":apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:"+
                ":apple:  :apple:  :apple:  :apple:")
    
    st.markdown("\U0001f956  \U0001f956  \U0001f956  \U0001f956  \U0001f956  \U0001f956  \U0001f956"+
                " \U0001f956  \U0001f956  \U0001f956  \U0001f956  \U0001f956  \U0001f956  \U0001f956"+
                " \U0001f956  \U0001f956  \U0001f956  \U0001f956  \U0001f956  \U0001f956  \U0001f956"+
                " \U0001f956  \U0001f956  \U0001f956  \U0001f956  \U0001f956  \U0001f956")

         
    st.markdown(":truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:"+
                " :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:"+
                " :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:"+
                " :truck:  :truck:  :truck:  :truck:  :truck:  :truck:")
             
    st.markdown("\U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2 \U0001f6D2"+
                " \U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2 \U0001f6D2"+
                " \U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2 \U0001f6D2"+
                " \U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2  \U0001f6D2")             
    #st.markdown("\U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D"+
    #            " \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D"+
    #            " \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D"+
    #            " \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D  \U0001f37D")
                
    #-------------------------------------------------------------------------#    
    st.header("Qu'est-ce qu'Agribalyse ?")   
    st.markdown("""
                [AGRIBALYSE](https://doc.agribalyse.fr/documentation/) est une
                base de données de référence des indicateurs d’impacts 
                environnementaux des produits agricoles produits en France 
                et des produits alimentaires consommés en France.   
                Il y a une vidéo de présentation dans le lien. 
    """)
     
    #-------------------------------------------------------------------------#
    st.header(" Que trouve-t-on dans ces données ? \U0001f5C4")
    st.markdown("""
                Elles recensent des caractéristiques de plusieurs aliments 
                ainsi que les émissions de polluants qui leur sont associés.
                """)
                
     
    #-------------------------------------------------------------------------#
    st.header("Que pouvez-vous faire avec cette petite application ? ")
    st.markdown("""
                Dans la version actuelle, elle vous permet 
                * d'explorer les
                données à l'aide essentiellement de Plotly  \U0001f642 (graphiques interactifs)
                * de construire des modèles pour prédire le DQR en fonction de différents indicateurs.
     
                """
                )
    
    #-------------------------------------------------------------------------#
    st.header("Qu'est-ce que le DQR ?")
    st.markdown("""
                Une note de qualité - le **Data Quality Ratio (DQR)** - de 1, très bon, à 5, 
                très mauvais - est associée à chaque produit agricole et alimentaire pour 
                lequel Agribalyse fournit des inventaires de cycle de vie et des 
                indicateurs d’impacts. La Commission Européenne recommande de la 
                prudence dans l’utilisation des données avec des DQR supérieurs à 3. 
                Dans la base de données AGRIBALYSE, 67 % des données ont un DQR jugé bon 
                ou très bon (1 à 3).
                """
                )
                
                
#==============================   Page 2  ====================================#
#========================  Exploration des données  ==========================#
def page2():
    
    
    st.sidebar.markdown("")
    
    #--Sélection du Thème des graphique----#
    theme_select = st.sidebar.selectbox("Choisissez le thème de vos graphiques pour la suite (il y a quelques conflits avec celui de Streamlit)" ,
                                        ['ggplot2', 'seaborn', 'simple_white', 'plotly',
                                         'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
                                         'ygridoff', 'gridon', 'none'])
    
    pio.templates.default = theme_select
    #-----------------------------------#    
    
    
    
    st.title('Exploration des données')
    st.write("##")
    
    #-------------------------------------------------------------------------#
    #-------------------------------------------------------------------------#
    st.header("Sources des données")
    st.markdown("""
                [lien 1](https://www.data.gouv.fr/fr/posts/les-donnees-ouvertes-pour-lapprentissage-automatique-machine-learning/) 
                \> [lien 2](https://datascience.etalab.studio/dgml/) 
                \> [lien 3](https://datascience.etalab.studio/dgml/c763b24a-a0fe-4e77-9586-3d5453c631cd) 
                \> [lien 4](https://www.data.gouv.fr/en/datasets/agribalyse-r-detail-par-ingredient/)
    """)    

    st.write("Il y a plusieurs fichiers, nous utilisons ici la 'Synthèse'.") 
       
    #-------------------------------------------------------------------------#    
    st.header("Données")
    
    synthese_dataset=pd.read_csv("datasets/Agribalyse_Synthese.csv", header=0)
    st.write(synthese_dataset)
    
    
    #*************************************************************************#
    #*************************************************************************#
    st.header("1e niveau d'exploration")
    
    #------------------------------------#
    st.subheader('Signification des variables')
    st.markdown("""
                
* La 1e partie des variables est plutôt explicite. 
* Pour les variables plus opaques, voir la [documentation](https://doc.agribalyse.fr/documentation/methodologie-acv).
* Voici quelques points :
    * **Score unique EF** 
        * EF = Environmental Footprint 
        * Le score unique EF est sans unité.
        * Plus le score est bas plus son impact sur l'environnement est faible. 
        * La valeur absolue n’est pas pertinente en tant que telle, l’intérêt est la comparaison entre produit.
        * Ce score unique est une moyenne pondérée des 16 indicateurs environnementaux.
    * Une note de qualité - le **Data Quality Ratio (DQR)** - de 1, très bon, à 5, 
    très mauvais - est associée à chaque produit agricole et alimentaire pour 
    lequel Agribalyse fournit des inventaires de cycle de vie et des 
    indicateurs d’impacts. La Commission Européenne recommande de la 
    prudence dans l’utilisation des données avec des DQR supérieurs à 3. 
    Dans la base de données AGRIBALYSE, 67 % des données ont un DQR jugé bon 
    ou très bon (1 à 3).
                """)
  
    #-------------------------------------------------------------------------#
    st.subheader('Informations générales')
    
    st.markdown("#### Format des données")
    st.write(synthese_dataset.shape)
    
    st.markdown("#### Identification des variables")
    st.markdown("Type des variables")
    st.write(synthese_dataset.dtypes)
    
    st.write("Variable cible :dart:")
    #target = st.selectbox("Quelle est votre variable cible ?", 
    #                      synthese_dataset.columns, 
    #                     index=11)   # variable affichée par defaut
    target = 'DQR - Note de qualité de la donnée (1 excellente ; 5 très faible)'
    st.markdown("La variable cible est '{}'.".format(target))
    
    
    st.markdown("#### Données manquantes")
    st.write("Il n'y a aucune donnée manquante.")
    st.write(pd.DataFrame(synthese_dataset.isna().sum()))
    
    
    
    
    #*************************************************************************#
    #*************************************************************************#
    st.header("2e niveau d'exploration")
    
    
    
    #-------------------------------------------------------------------------#
    st.subheader("Analyse univariée")
    
    #-------------------------------------------------------------------------#
    st.markdown("#### Variable continue")
    
    #-----------------------------------#
    st.markdown("##### Histogramme")
    
    var_cont = st.selectbox("Sélectionnez une variable continue", 
                            synthese_dataset.select_dtypes(float).columns,
                            index=0)   # variable affichée par defaut
    
    number_bins = st.slider('Nombre de bins', min_value=4, max_value=500,
                            value=4)
    
    fig = px.histogram(synthese_dataset, x=var_cont, nbins=number_bins,
                       histnorm='probability',
                       marginal='box',
                       title='Histogramme de la variable {}'.format(var_cont),
                       color_discrete_sequence=["#748726"])
    st.write(fig)
    
    #-----------------------------------#
    st.markdown("##### Coefficient d'asymétrie")
    var_conts_skew = st.multiselect("Sélectionnez les variables continues", 
                                    synthese_dataset.select_dtypes(float).columns,
                                    key='cont_skewness')  
    
    if var_conts_skew != []:
        skew_list=[]
        for col in var_conts_skew:
            skew_list.append(skew(synthese_dataset[col]))
    
        st.write(pd.DataFrame(skew_list, index=var_conts_skew))
    
    #-----------------#
    with st.beta_expander("Compléments"):
        st.markdown("""
                    Lien : [documentation scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html)
                    """)
        st.markdown("") 
        st.markdown("Formule :")
        
        st.latex(r'''
                 \beta_1 = \frac{1}{I} \sum_{i} (\frac{x_i - \bar{x}}{s})^3
                 ''')
        st.markdown("""
                    avec :
                    * I = nombre d'observations
                    * s : l'écart-type.
                    """) 
        st.markdown("") 
        st.markdown(""" 
                    Idées :  
                    * on prend en compte les écarts à la moyenne
                    * on se ramène à un écart-type de 1
                    * l'élévation au cube conserve le signe des écarts et fait jouer un grand rôle aux valeurs extrêmes.
                    """)   
        st.markdown("") 
        st.markdown(""" 
                    Interprétation :   
                    * si le coefficient est > 0, cela indique une distribution étalée vers la droite, avec la présence d'une queue de distribution étalée vers la droite. 
                    * si le coefficient est < 0, cela indique une distribution étalée vers la gauche, avec la présence d'une queue de distribution étalée vers la gauche. 
                    """)   
    #-----------------#
    
    
    
    
    
    
    #-------------------------------------------------------------------------#
    st.markdown("#### Variable catégorielle")
    var_cat = st.selectbox("Sélectionnez une variable catégorielle", 
                              synthese_dataset.select_dtypes(object).columns,
                              key="cat",
                              index=1)   # variable affichée par defaut
    
    fig = px.histogram(synthese_dataset, x=var_cat,
                       title='Distribution de la variable {}'.format(var_cat),
                       color_discrete_sequence=["#748726"])
    fig.update_xaxes(categoryorder="total descending")
    st.write(fig)


    
    #-------------------------------------------------------------------------#
    st.subheader("Analyse bivariée")
    
    st.markdown("On analyse ici les relations avec la variable cible.") 



    #-------------------------------------------------------------------------#
    st.markdown("#### Target et Variable continue")
    
    #-----------------------------------#
    #-----------------------------------#
    st.markdown("##### Graphiques")
    
    # choisir le type de graphique
    type_cont = st.selectbox("Sélectionnez le type de graphique", 
                              ["Scatter Matrix", "Parallel Coordinates Plot", "(option) Parallel Coordinates Plot"], 
                              index=0)    # variable affichée par defaut
    
    #-----------------------------------#
    if type_cont=="Scatter Matrix":
        var_cont_scatter_mat = st.selectbox("Sélectionnez une variable continue", 
                                            synthese_dataset.select_dtypes(float).columns,
                                            key="target_cont_scatter_matr", 
                                            index=1)  # valeur par défaut
        if var_cont_scatter_mat !=():
            fig = px.scatter_matrix(synthese_dataset,
                                    dimensions=[target,var_cont_scatter_mat], 
                                    title='Scatter Matrix',
                                    color_continuous_scale=px.colors.diverging.Fall,
                                    color_discrete_sequence=px.colors.qualitative.Antique)
            # taille markers
            fig.update_traces(marker=dict(size=1))
            # enlever ou réduire les labels, valeurs, ticks qui rendent illisibles
            nb_col=len(var_cont_scatter_mat)  
            fig.update_layout({"xaxis" + str(i+1): dict(showticklabels=False, ticklen=0, titlefont=dict(size=(nb_col+3.8)/nb_col)) for i in range(nb_col)})
            fig.update_layout({"yaxis" + str(i+1): dict(showticklabels=False, ticklen=0, titlefont=dict(size=(nb_col+2.2)/nb_col)) for i in range(nb_col)})
            fig.update_layout(autosize=False, width=750, height=710)
            st.write(fig)
    
            # st.markdown("""*Mmm ... Petit problème de labels à régler :confused:*""")
   
    #-----------------------------------#
    elif type_cont=="Parallel Coordinates Plot":
        var_cont_parallel = st.selectbox("Sélectionnez une variable continue", 
                                         synthese_dataset.select_dtypes(float).columns, 
                                         key="cont_parallel", 
                                         index=1)    # valeur par défaut
        if var_cont_parallel !=[]:
            fig = px.parallel_coordinates(synthese_dataset,
                                          dimensions=[target,var_cont_parallel], 
                                          title='Parallel Coordinates Chart',
                                          color_continuous_scale=px.colors.diverging.Fall
                                          )
            st.write(fig)
           
    
    #-----------------------------------#
    elif type_cont=="(option) Parallel Coordinates Plot":
        var_conts_parallel = st.multiselect("Sélectionnez des variables continues", 
                                            synthese_dataset.select_dtypes(float).columns, 
                                            key="cont_parallel"
                                            )   
        if var_conts_parallel !=[]:
            # changer les noms des labels trop longs
            labels_long={"DQR - Note de qualité de la donnée (1 excellente ; 5 très faible)":"DQR",
                         "Score unique EF (mPt/kg de produit)" : "Score unique EF"}
            
            
            fig = px.parallel_coordinates(synthese_dataset,
                                          dimensions=var_conts_parallel, 
                                          title='Parallel Coordinates Chart',
                                          color_continuous_scale=px.colors.diverging.Fall,
                                          labels=labels_long
                                          )
            st.write(fig)
    
    #-----------------------------------#        
    #-----------------------------------#
    st.markdown("##### Coefficient de corrélation")
    
    st.write(pd.DataFrame(synthese_dataset.corr()[target].sort_values(ascending=False)))
           
    
    
    #-------------------------------------------------------------------------#
    st.markdown("#### Target et Variable Catégorielle")
    
    
    
    #st.markdown(" *Malgré le message d'erreur quand le multiselect est vide, cela semble fonctionner ...  :confused: *")
    
    # choisir le type de graphique  
    type_cont_cat = st.selectbox("Sélectionnez le type de graphique", 
                                 ["Box plot", "Ridgeline", "Parallel categories plot",
                                  "(option, non stable) Treemap"],
                                 key="cont_cat")
   
    
    #-----------------------------------#
    if type_cont_cat=="Box plot":
        cat2_box = st.selectbox("Sélectionnez une variable catégorielle ('Code AGB', 'Nom du Produit en Français', 'LCI Name' ont été supprimées)", 
                                synthese_dataset.select_dtypes(object).drop(['Code AGB', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                key="cat2_box")
        if cat2_box != []:
            fig = px.box(synthese_dataset, 
                         x=cat2_box, 
                         y=target,
                         color=cat2_box,
                         title="Box plot",
                         color_discrete_sequence=px.colors.qualitative.Antique)
            fig.update_layout(boxgap=0, showlegend=False)
            fig.update_xaxes(tickangle=45)
            st.write(fig) 
    
    #-----------------------------------#
    elif type_cont_cat=="Ridgeline":
        cat2_ridge = st.selectbox("Sélectionnez une variable catégorielle ('Code AGB', 'Nom du Produit en Français', 'LCI Name' ont été supprimées)", 
                                  synthese_dataset.select_dtypes(object).drop(['Code AGB', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                  key="cat2_ridge")
        if cat2_ridge !=[]:
            fig = px.violin(synthese_dataset, 
                            x=target, 
                            y=cat2_ridge,
                            orientation='h', 
                            color=cat2_ridge,
                            title="Ridgeline".format(target, cat2_ridge),
                            color_discrete_sequence=px.colors.qualitative.Antique)
            fig.update_traces(side='positive', width=2)
            fig.update_layout(showlegend=False) 
            st.write(fig)  
 
    #-----------------------------------#
    elif type_cont_cat=="Parallel categories plot":
        cat2_parallel = st.selectbox("Sélectionnez une variable catégorielle ('Code AGB', 'Nom du Produit en Français', 'LCI Name' ont été supprimées)", 
                                      synthese_dataset.select_dtypes(object).drop(['Code AGB', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                      key="cat2_parallel")  
        
        if cat2_parallel !=[]:
            fig = px.parallel_categories(synthese_dataset, dimensions=[cat2_parallel], 
                                         color=target, 
                                         color_continuous_scale=px.colors.diverging.Fall, 
                                         color_continuous_midpoint=3)
            st.write(fig)
    
    #-----------------------------------#
    elif type_cont_cat=="(option, non stable) Treemap":
        var_cat1_cat2_tree = st.multiselect("Sélectionnez deux variables catégorielles", 
                                            synthese_dataset.select_dtypes(object).drop(['Code AGB', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                            key="cat1_cat2_tree"
                                            )
        if var_cat1_cat2_tree != []:
            fig = px.treemap(synthese_dataset,
                             path=var_cat1_cat2_tree, 
                             values=target,
                             color=target,
                             color_continuous_scale=px.colors.diverging.Fall,
                             color_continuous_midpoint=3,
                             title="Treemap (proportions et couleurs de la variable cible)")
            st.write(fig)
            
            
            
            
            
    #*************************************************************************#
    #*************************************************************************#
    st.header("3e niveau d'exploration")
    
            
    st.markdown("Nous analysons les corrélations possibles entre les variables sans la variable cible.")
            
    #-------------------------------------------------------------------------#
    st.subheader(" Entre variables numériques")        
   
    # colonnes type 'float' sans la variable cible
    col_float_no_target = synthese_dataset.select_dtypes(float).drop(target,axis=1).columns
    nb_col = len(col_float_no_target) 
    
    
    #-----------------------------------#
    select_conts = st.selectbox("Sélectionnez le type de graphique",
                                ["Scatter Matrix complète", "Scatter Matrix", "Clustermap", "Parallel Coordinates Plot"],
                                key="select_conts") 
    
    
    #-----------------------------------#
    if select_conts=="Scatter Matrix complète":
        fig = px.scatter_matrix(synthese_dataset, dimensions=col_float_no_target,
                                title="Scatter matrix des variables numériques, sans la variable cible",
                                color_continuous_scale=px.colors.diverging.Fall,
                                color_discrete_sequence=px.colors.qualitative.Antique)
        fig.update_traces(marker=dict(size=1))
        fig.update_layout({"xaxis" + str(i+1): dict(showticklabels=False, ticklen=0, titlefont=dict(size=(nb_col+3.8)/nb_col)) for i in range(nb_col)})
        fig.update_layout({"yaxis" + str(i+1): dict(showticklabels=False, ticklen=0, titlefont=dict(size=(nb_col+2.2)/nb_col)) for i in range(nb_col)})
        fig.update_layout(autosize=False, width=750, height=710)
        st.write(fig)
            
    #-----------------------------------#
    elif select_conts=="Scatter Matrix":
         var_conts_scatter_mat = st.multiselect("Sélectionnez les variables continues", 
                                                synthese_dataset.select_dtypes(float).drop(target,axis=1).columns,
                                                key="conts_scatter_matr")
         if var_conts_scatter_mat !=():
             fig = px.scatter_matrix(synthese_dataset,
                                     dimensions=var_conts_scatter_mat, 
                                     title='Scatter Matrix',
                                     color_continuous_scale=px.colors.diverging.Fall,
                                     color_discrete_sequence=px.colors.qualitative.Antique)
             # taille markers
             fig.update_traces(marker=dict(size=2.5))
             # enlever ou réduire les labels, valeurs, ticks qui rendent illisibles
             nb_col=len(var_conts_scatter_mat)  
             fig.update_layout({"xaxis" + str(i+1): dict(showticklabels=False, ticklen=0, titlefont=dict(size=(nb_col+4)/nb_col)) for i in range(nb_col)})
             fig.update_layout({"yaxis" + str(i+1): dict(showticklabels=False, ticklen=0, titlefont=dict(size=(nb_col+3)/nb_col)) for i in range(nb_col)})
             #fig.update_layout({"xaxis" : dict( titlefont=dict(size=(nb_col+3.8)/nb_col)) })
             #fig.update_layout({"yaxis" : dict( titlefont=dict(size=(nb_col+2.2)/nb_col)) })             
             fig.update_layout(autosize=False, width=750, height=710)
             st.write(fig)
    
            # st.markdown("""*Mmm ... Petit problème de labels à régler :confused:*""")
   
    
    #-----------------------------------#
    elif select_conts=="Clustermap":
        fig = plt.figure()
        fig = sns.clustermap(synthese_dataset.select_dtypes(float).drop(target,axis=1).corr(),
                             cmap="vlag",
                             figsize=(15,15),
                             cbar_pos=(0.00, 0.9, 0.05, 0.18))
        st.pyplot(fig)
        
        
    #-----------------------------------#
    elif select_conts=="Parallel Coordinates Plot":
        
        var_conts_parallel = st.multiselect("Sélectionnez les variables continues (2 ou plus)", 
                                            synthese_dataset.select_dtypes(float).drop(target,axis=1).columns, 
                                            key="conts_parallel")
        if var_conts_parallel !=[]:
            fig = px.parallel_coordinates(synthese_dataset,
                                          dimensions=var_conts_parallel, 
                                          title='Parallel Coordinates Chart',
                                          color_continuous_scale=px.colors.diverging.Fall)
            st.write(fig)
        
        
        
        
        
    #-------------------------------------------------------------------------#         
    st.subheader(" Entre variables catégorielles")
    
    # colonnes type 'object' sans la variable cible, sans 'Nom du Produit en Français'
    col_object_no_target = synthese_dataset.select_dtypes(object).drop('Nom du Produit en Français', axis=1).columns
    
    
    #-----------------------------------#
    select_cats = st.selectbox("Sélectionnez l'outil",
                              ["Histogramme", "Table de contingence"],
                              key="select_cats") 
    
    
    #-----------------------------------#
    if select_cats=="Histogramme":
        var_cats_hist = st.multiselect("Sélectionnez les variables catégorielles ('Code AGB', 'Nom du Produit en Français', 'LCI Name' ont été supprimées)", 
                                       synthese_dataset.select_dtypes(object).drop(['Code AGB', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                       key="cats_hist")
        
        if var_cats_hist != []:
            cat1_h = var_cats_hist[0]
            cat2_h = var_cats_hist[1]
            fig = px.histogram(synthese_dataset, x=cat1_h, color=cat2_h, 
                               color_discrete_sequence=px.colors.qualitative.Antique)            
            st.write(fig)
    
    
    #-----------------------------------#
    elif select_cats=="Table de contingence":
        var_cats_table = st.multiselect("Sélectionnez les variables catégorielles ('Code AGB', 'Nom du Produit en Français', 'LCI Name' ont été supprimées)", 
                                        synthese_dataset.select_dtypes(object).drop(['Code AGB', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                        key="cats_table")

        if var_cats_table != []:
            cat1_t = var_cats_table[0]
            cat2_t = var_cats_table[1]
            
            # table de contingence
            df = synthese_dataset[[cat1_t,cat2_t]].pivot_table(index=cat1_t,columns=cat2_t,
                                                             aggfunc=len,
                                                             # margins=True,margins_name="Total",
                                                             fill_value=0)   # remplacer les Nan par 0 
            # heatmap
            #fig = plt.figure(figsize=(22,8))
            # fig = sns.heatmap(df)   # d=integer
            
            
            fig, ax = plt.subplots()
            sns.heatmap(df, ax=ax, annot=True, fmt='d')   # d=integer
            st.write(fig)
  
            st.markdown("""*Mmm ... Affichage à améliorer :confused:*""")
    
  

    #-------------------------------------------------------------------------#
    st.subheader(" Entre variables numériques et catégorielles")        
                
    
    # colonnes type 'object' sans la variable cible, sans 'Nom du Produit en Français'
    col_object_no_target = synthese_dataset.select_dtypes(object).drop('Nom du Produit en Français', axis=1).columns
    # colonnes type 'float' sans la variable cible
    col_float_no_target = synthese_dataset.select_dtypes(float).drop(target, axis=1).columns
    
    
    
    #-----------------------------------#
    # choisir le type de graphique  
    type_cont_cat = st.selectbox("Sélectionnez le type de graphique", 
                                 ["Box plot", "Ridgeline", "Parallel categories plot", "(option) Treemap"],
                                 key="cont_cat")
   
   
    
    #-----------------------------------#
    if type_cont_cat=="Box plot":
        var_cont1_cat2_box = st.multiselect("Sélectionnez une variable continue, puis une variable catégorielle ('Code AGB', 'Code CIQUAL', 'Nom du Produit en Français', 'LCI Name' ont été supprimées)", 
                                            synthese_dataset.drop([target,'Code AGB', 'Code CIQUAL', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                            key="cont_1_cat2_box")
        if var_cont1_cat2_box != []:
            var_cont1_box=var_cont1_cat2_box[0]
            var_cat2_box=var_cont1_cat2_box[1]
            fig = px.box(synthese_dataset, 
                         x=var_cat2_box, 
                         y=var_cont1_box,
                         color=var_cat2_box,
                         title="{} en fonction de {}".format(var_cont1_box, var_cat2_box),
                         color_discrete_sequence=px.colors.qualitative.Antique)
            fig.update_layout(boxgap=0, showlegend=False)
            fig.update_xaxes(tickangle=45)
            st.write(fig) 
            
            
    #-----------------------------------#
    elif type_cont_cat=="Ridgeline":
        var_cont1_cat2_ridge = st.multiselect("Sélectionnez une variable continue, puis une variable catégorielle ('Code AGB', 'Code CIQUAL', 'Nom du Produit en Français', 'LCI Name' ont été supprimées)", 
                                              synthese_dataset.drop([target,'Code AGB', 'Code CIQUAL', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                              key="cont_1_cat2_ridge")
        if var_cont1_cat2_ridge !=[]:
            var_cont1_ridge=var_cont1_cat2_ridge[0]
            var_cat2_ridge=var_cont1_cat2_ridge[1]
            fig = px.violin(synthese_dataset, 
                            x=var_cont1_ridge, 
                            y=var_cat2_ridge,
                            orientation='h', 
                            color=var_cat2_ridge,
                            title="{} en fonction de {}".format(var_cont1_ridge, var_cat2_ridge),
                            color_discrete_sequence=px.colors.qualitative.Antique)
            fig.update_traces(side='positive', width=2)
            fig.update_layout(showlegend=False) 
            st.write(fig)  
            
            
            
    #-----------------------------------#
    elif type_cont_cat=="Parallel categories plot":
        var_cont1_cat2_parallel = st.multiselect("Sélectionnez une variable continue, puis une variable catégorielle ('Code AGB', 'Code CIQUAL', 'Nom du Produit en Français', 'LCI Name' ont été supprimées)", 
                                                 synthese_dataset.drop([target,'Code AGB', 'Code CIQUAL', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                                 key="cont_1_ca2_parallel")  
        
        if var_cont1_cat2_parallel !=[]:
            var_cont1_parallel=var_cont1_cat2_parallel[0]
            var_cat2_parallel=var_cont1_cat2_parallel[1]
            
            fig = px.parallel_categories(synthese_dataset, dimensions=[var_cat2_parallel], 
                                         color=var_cont1_parallel, 
                                         color_continuous_scale=px.colors.diverging.Fall, 
                                         color_continuous_midpoint=3)
            st.write(fig)
          
            
    #-----------------------------------#
    elif type_cont_cat=="(option) Treemap":
        var_cont1_cat2_cat3_tree = st.multiselect("Sélectionnez une variable continue, puis deux variables catégorielles ('Code AGB', 'Code CIQUAL', 'Nom du Produit en Français', 'LCI Name' ont été supprimées)", 
                                                  synthese_dataset.drop([target,'Code AGB', 'Code CIQUAL', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                                  key="cont_1_cat2_cat_3_tree")
        
        if var_cont1_cat2_cat3_tree !=[]:
            cont1_tree = var_cont1_cat2_cat3_tree[0]
            cat2_tree = var_cont1_cat2_cat3_tree[1]
            cat3_tree = var_cont1_cat2_cat3_tree[2]

            fig = px.treemap(synthese_dataset,
                             path=[cat2_tree,cat3_tree],
                             values=cont1_tree,
                             color=cont1_tree,
                             title="Treemap (proportions et couleurs avec la variable continue)",
                             color_continuous_scale=px.colors.diverging.Fall
                             )
            st.write(fig) 
            
            st.markdown("""*Mmm ... Valeurs bizarres, paramétrage à vérifier :confused:*""")
    
    
    #*************************************************************************#
    #*************************************************************************#
    st.header("4e niveau d'exploration")
    
    st.markdown("Après une analyse essentiellement graphique, nous complétons par quelques statistiques. ")
  
    
    #-------------------------------------------------------------------------#
    st.subheader(" Entre variables numériques et catégorielles")    
    
    st.markdown("#### Indicateur $\eta^2$")
  
    st.markdown("""
    On utilise un indicateur appelé le **rapport de corrélation** $\eta^2$ :  
    * c'est un nombre compris entre 0 et 1
    * si $\eta^2$=0, il n’y a pas a priori de relation entre les variables
    * si $\eta^2$=1, il n’existe pas a priori de relation entre les variables.
    """)
    
    with st.beta_expander("Compléments"):
        st.markdown("""
                    [Formules](https://openclassrooms.com/fr/courses/4525266-decrivez-et-nettoyez-votre-jeu-de-donnees/4774896-analysez-une-variable-quantitative-et-une-qualitative-par-anova) :  
                        """)
    
        st.latex(r'''
             SCT = 
             \sum_{j} (y_{j} - \bar{y})^2
             ''')
        st.latex(r'''
             SCE = 
             \sum_j n_j (\bar{y_j} - \bar y)^2
             ''')                 
        st.latex(r''' 
             \eta^2 =
             \frac{SCE}{SCT}
             ''')
    
    
    #-----------------------------------#
    #Fonction pour calculer $\eta^2$
    #On commence par définir une fonction pour calculer eta^2.
    
    # x : variable catégorielle 
    # y : variable quantitative

    def eta_squared(x,y):
        # moyenne de y (y^bar)
        moyenne_y = y.mean()
    
        # on récupère dans une liste les informations sur les classes sous forme de dictionnaire :
            # taille de la classe (n_i) et moyenne de la classe (yi^bar)
        classes = []
    
        # on fait une boucle sur les classes
        # pour chaque classe :
            # récupérer les valeurs de y relative à la classe ()
            # récupérer la taille de la classe (n_i) et la moyenne de la classe (y_i^bar)
    
        for classe in x.unique():
            yi_classe = y[x==classe]
            classes.append({'ni': len(yi_classe),
                            'moyenne_classe': yi_classe.mean()})
        
        # on calcule :
            # la variation totale SCT
            # la variation interclasse SCE
        # on retourne le rapport de corrélation eta^2
        SCT = sum([(yj-moyenne_y)**2 for yj in y])
        SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
        return SCE/SCT
    
    
    #-----------------------------------#
    st.markdown("#### Application")
    st.markdown("")
    
    var_cont1_cat2_eta2 = st.multiselect("Sélectionnez une variable continue, puis une variable catégorielle ('Code AGB', 'Code CIQUAL', 'Nom du Produit en Français', 'LCI Name' ont été supprimées)", 
                                         synthese_dataset.drop(['Code AGB', 'Code CIQUAL', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                         key="cont_1_cat2_eta2") 

    if var_cont1_cat2_eta2 !=[]:
        cont1_eta = var_cont1_cat2_eta2[0]
        cat2_eta = var_cont1_cat2_eta2[1]
        st.write("{} et {} : \n{:.2f}".format(cat2_eta,cont1_eta, eta_squared(synthese_dataset[cat2_eta], synthese_dataset[cont1_eta])))
    
    


    #-------------------------------------------------------------------------#
    st.subheader("Entre variables catégorielles")    
    
    st.markdown("#### Table de contingence avec indicateur de dépendance")
  
    st.markdown("")
    st.markdown("""
    Ici, on ne peut pas utiliser le test du khi2 vu que la condition asymptotique n'est pas vérifiée (ie : les effectifs ne sont pas tous supérieurs à 5).  
    Nous allons colorer les cases non pas avec l'effectif mais avec une valeur qui donne une indication sur l'indépendence des deux variables.   
    
    Les idées :
    * on mesure l'écart entre l'effectif observée (une cellule de la table) et l'effectif attendu en cas d'indépendance (produit des deux totaux divisé par l'effectif total) 
    * on transforme pour avoir un nombre entre 0 et 1 
    * on peut voir ce nombre comme une contribution à la non-indépendance des deux variables 
    * plus la case est claire, plus la case est source de non-indépendance.   
    """)
    
    with st.beta_expander("Compléments"):
        st.markdown("""
    [Formules](https://openclassrooms.com/fr/courses/4525266-decrivez-et-nettoyez-votre-jeu-de-donnees/4775616-analysez-deux-variables-qualitatives-avec-le-chi-2) :
                """) 
                
    #st.latex(r'''
    #         une cellule de la table $n_{ij}
    #         ''')
    #st.latex(r'''
    #         \frac{n_{i.} n_{.j}}{n} 
    #         ''')
            
    #st.latex(r'''
    #         n_{ij} - \frac{n_{i.} n_{.j}}{n}
    #         ''')

    #st.latex(r'''
    #        (n_{ij} - \frac{n_{i.} n_{.j}}{n})^2
    #         ''')
            
        st.latex(r'''
                 \xi_{ij} = \frac{(n_{ij} - \frac{n_{i.} n_{.j}}{n})^2}{ \frac{n_{i.} n_{.j}}{n}}
                 ''')
            
        st.latex(r'''
                 \xi_n = \sum_i \sum_j\xi_{ij}
                 ''')
            
            
        st.latex(r'''
                 {\xi_{ij}}_{normalisé} = \frac {\xi_{ij}}{\xi_n}
                 ''')
            
  
    #-----------------------------------#
    #Fonction pour construire la table avec indicateur de dépendance
    
    # construisons d'abord une fonction 
    # qui prend : (1) le dataset (2) les deux variables catégorielles
    # et retourne : (1) la table de contingence (2) la table des xi_{ij} normalisée

    def table_xi_normalisee(dataset, var1, var2):
        # table de contingence  
        # on rajoute totaux 
        # on remplace les NaN par 0
        cont = dataset[[var1, var2]].pivot_table(index=var1, columns=var2, 
                                                 aggfunc=len,
                                                 margins=True, margins_name="Total").fillna(0) 

        tx = cont.loc[:,["Total"]]         # total en colonne
        ty = cont.loc[["Total"],:]         # total\sum_j (\hat{y_j} - \bar y)^2 en ligne
        n = len(dataset)                   # effectif total
        indep = tx.dot(ty) / n             # produit des deux totaux divisé par l'effectif total  𝑛_𝑖. * 𝑛_.𝑗 / 𝑛  
        
        measure = (cont-indep)**2/indep    # xi_{ij}
        xi_n = measure.sum().sum()
        table = measure/xi_n               # xi_{ij} normalisé
    
        return cont, table

        # notons qu'on obtient des couleurs identiques que l'on prenne \x_ij ou la version normalisée
         
    
    #-----------------------------------#
    st.markdown("#### Application")
    st.markdown("")
    
    var_cat1_cat2_xi = st.multiselect("Sélectionnez deux variables catégorielles( 'Code AGB', 'Nom du Produit en Français', 'LCI Name' ont été supprimées)", 
                                         synthese_dataset.select_dtypes(object).drop(['Code AGB', 'Nom du Produit en Français', 'LCI Name'], axis=1).columns,
                                         key="cat_1_cat2_xi") 
    
    if var_cat1_cat2_xi != []:
            cat1_xi = var_cat1_cat2_xi[0]
            cat2_xi = var_cat1_cat2_xi[1]
            # récupérer la table de contingence et la table des xi_ij normalisée avec la fonction précédente
            cont, table = table_xi_normalisee(synthese_dataset, cat1_xi, cat2_xi)
            # afficher avec une heatmap 
            fig, ax = plt.subplots()
            sns.heatmap(table.iloc[:-1,:-1], ax=ax, annot=cont.iloc[:-1,:-1], fmt='.0f')    # on affiche toujours les effectifs, pas l'indicateur, qui est représenté par la couleur 
                                                                                  # on enlève les colonnes des totaux
            st.write(fig)
            
            st.markdown("""*Mmm ... Affichage à améliorer :confused:*""")
    
  
    
  
    
  
    
  
  
    
#==============================   Page 2_1  ===================================#
#==================  Explorations : Réduction de dimension  ===================#
def page2_1(): 
    st.title("Exploration de données : Réduction de dimension")
    
    
    st.markdown("""
                Voici quelques outils supplémentaires pour analyser les données. 
                """)
      
    #*************************************************************************#
    #*************************************************************************#
    st.header("Données")
    
    data_original = pd.read_csv("datasets/Agribalyse_Synthese.csv", header=0)
    st.write(data_original)
  
    
    #*************************************************************************#
    #*************************************************************************#
    st.markdown("")
    st.markdown("")
    st.header("Préparation des données")
    st.markdown("Il faut préparer les données un minimum pour pourvoir appliquer certains algorithmes.")
    
    #----------------------------------#
    # Créer une copie pour les modifications
    data_original_copy = data_original.copy()


    
    #-------------------------------------------------------------------------#
    st.subheader("Feature selection simple 🗑️")
    
    var_to_delete_simple = st.multiselect("Sélectionnez les variables à supprimer", 
                                          data_original.columns,
                                          key="var_to_delete_simple ") 
    if var_to_delete_simple !=[]:
        data_original_copy = data_original_copy.drop(var_to_delete_simple, axis=1)
        st.write(data_original_copy)

    #-------------------------------------------------------------------------#
    st.subheader("Encodage des variables catégorielles")
        
    encoding_mth = st.selectbox("Sélectionnez la méthode d'encodage", 
                                ["Label Encoding", "One-Hot Encoding","Binary Encoding"],
                                key="encoding_mth") 
    
    
    if encoding_mth != None:
        # liste des colonnes type 'object'
        col_object = data_original_copy.select_dtypes(object).columns
            
        if encoding_mth=="Label Encoding":
            # créer l'encodeur
            label_encoder = LabelEncoder()
            for col in col_object:
                data_original_copy[col] = label_encoder.fit_transform(data_original_copy[col])
        
            # afficher le nouveau dataset
            st.markdown("*Données après Label Encoding*")
            st.write(data_original_copy)
            
        elif encoding_mth=="One-Hot Encoding":
            # créer l'encodeur
            OH_encoder = OneHotEncoder(sparse=False)
                
            # appliquer l'encodeur : cela retourne un array
            OH_array = OH_encoder.fit_transform(data_original_copy[col_object])
            # transformer en dataframe + rajouter les noms de colonnes
            OH_df = pd.DataFrame(OH_array)
            # remettre les bons index
            OH_df.index = data_original_copy.index
            # supprimer les colonnes 'object' du dataset initial
            df_initial_num = data_original_copy.drop(col_object, axis=1)
            # concaténer les deux dataframe
            data_original_copy = pd.concat([df_initial_num,OH_df], axis=1)
            # afficher le nouveau dataset
            st.markdown("*Données après One-Hot Encoding*")
            st.write(data_original_copy)
            
        elif encoding_mth=="Binary Encoding":
            # créer l'encodeur : on précise les colonnes à encoder
            binary_encoder = ce.BinaryEncoder(cols=col_object)
            # appliquer l'encodeur à nos données
            data_original_copy = binary_encoder.fit_transform(data_original_copy)
            # afficher le nouveau dataset
            st.markdown("*Données après Binary Encoding*")
            st.write(data_original_copy)
      
            
    #-------------------------------------------------------------------------#
    
    st.subheader("Feature Engineering simple")
     
    # colonnes à standardiser
    standard_columns_pca = st.multiselect("Sélectionner les colonnes à centrer-réduire",
                                          data_original_copy.columns,
                                          key='standard_columns_pca')
                       
    if standard_columns_pca !=[]:
        # définition du StandardScaler
        scaler = StandardScaler()
        # on applique le scaler aux colonnes concernées : cela retourne un array
        # on le transforme en un dataframe, en rajoutant les noms de colonne perdus
        # on récupère les bons index (ceux de X_train)
        standard_array = scaler.fit_transform(data_original_copy[standard_columns_pca])
        df_standardized = pd.DataFrame(standard_array, columns=standard_columns_pca)
        df_standardized.index = data_original_copy.index
        # on remplace les colonnes concernées par le dataframe précédent 'df_standardized'
        data_original_copy[standard_columns_pca] = df_standardized
        # afficher les nouvelles données
        st.write(data_original_copy)
                               
            
    #-------------------------------------------------------------------------#
    st.subheader("Données manquantes")
    
    st.write("Il n'y a aucune donnée manquante.")
    st.write(pd.DataFrame(data_original_copy.isna().sum()))
   
    
    
    
        
    
  
    #*************************************************************************#
    #*************************************************************************#
    st.markdown("")
    st.markdown("")
    st.header("Analyse en Composantes Principales (ACP)")
    st.markdown("Il s'agit d'une méthode de réduction de dimension.")
    
    
    #-------------------------------------------------------------------------#
    st.markdown("### Visualisation en 2D")
    
    #------------------------------------#
    # on définit un modèle de PCA
    pca = PCA(n_components=2)
    # on réduit le nombre de dimension avec le PCA
    X_proj = pca.fit_transform(data_original_copy)

    #------------------------------------#
    # dictionnaire pour les labels pour le scatter plot
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_* 100)
        }

    # variances expliquées : pour le titre
    var_comp1 = pca.explained_variance_ratio_[0]*100
    var_comp2 = pca.explained_variance_ratio_[1]*100
    var_expl_1_2 = var_comp1 + var_comp2

    #-------------------------------------#
    st.markdown("#### Nuage des variables")
    
    # composantes principales
    pc = pca.components_.T

    # afficher le nuage des variables
    fig = px.scatter()
    
    ## étendue des axes 
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(range=[-1, 1])
    
    ## noms des axes
    fig.update_xaxes(title="PC1")
    fig.update_yaxes(title="PC2")
    
    ## on fait une boucle sur toutes les variables
    ## à chaque fois, on affiche une ligne et le nom de la variable 
    for i, feature in enumerate(data_original_copy.columns):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=pc[i, 0],
            y1=pc[i, 1],
            line=dict(color="#748726",width=2)
            )
        
        fig.add_annotation(
            x=pc[i, 0],
            y=pc[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
            )   
    st.write(fig)
  
   
    #-------------------------------------#
    st.markdown("#### Nuage des individus")
    
    # variable à utiliser pour la couleur
    var_for_color_pca_2D = st.selectbox("Choisir une variable pour colorer les points", 
                                        data_original_copy.columns,
                                        key="var_for_color_pca_2D"
                                        )
    
    # afficher le nuage des individus 
    fig = px.scatter(X_proj, x=0, y=1,
                     labels=labels,
                     color=data_original[var_for_color_pca_2D],    # on colorie avec les valeurs des données non standardisées
                     color_continuous_scale=px.colors.diverging.Fall,
                     #color_discrete_sequence=px.colors.qualitative.Antique,
                     title=f'Variance expliquée: {var_expl_1_2:.1f}%'
                     )
    fig.update_traces(marker=dict(size=4),
                      opacity=0.4
                      ) 
    st.write(fig)
        
    
        
    #-------------------------------------------------------------------------#
    st.markdown("### Visualisation en 3D")
    
    #------------------------------------#
    # on définit un modèle de PCA
    pca = PCA(n_components=3)
    # on réduit le nombre de dimension avec le PCA
    X_proj = pca.fit_transform(data_original_copy)

    
    #------------------------------------#
    # dictionnaire pour les labels pour le scatter plot
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_* 100)
        }

    # variances expliquées : pour le titre
    var_comp1 = pca.explained_variance_ratio_[0]*100
    var_comp2 = pca.explained_variance_ratio_[1]*100
    var_comp3 = pca.explained_variance_ratio_[2]*100
    var_expl_1_2_3 = var_comp1 + var_comp2 + var_comp3

   
    #-------------------------------------#
    st.markdown("#### Nuage des individus")
    
    # variable à utiliser pour la couleur
    var_for_color_pca_3D = st.selectbox("Choisir une variable pour colorer les points", 
                                        data_original_copy.columns,
                                        key="var_for_color_pca_3D"
                                        )
    
    # afficher le scatter plot
    fig = px.scatter_3d(X_proj, x=0, y=1, z=2,
                        labels=labels,
                        color=data_original[var_for_color_pca_3D],  # on colorie avec les valeurs des données non standardisées
                        color_continuous_scale=px.colors.diverging.Fall,
                        title=f'Variance expliquée: {var_expl_1_2_3:.1f}%'
                        )
    fig.update_traces(marker=dict(size=7),
                      opacity=0.4
                      ) 
    st.write(fig)
    
            
            
        
     
        

    
  
    
  
    
  
    
  
#==============================   Page 3  ===================================#
#================== Prédiction du DQR : Premiers modèles ====================#

def page3():
    
    st.title("Prédiction du DQR : Premiers modèles")

    #-------------------------------------------------------------------------#    
    st.header("Données originales")
    
    @st.cache(persist=True)
    def load_data_bis():
        data_original = pd.read_csv("datasets/Agribalyse_Synthese.csv", header=0)
        return data_original
    
    data_original=load_data_bis()
    st.write(data_original)
    
    #----------------------------------#
    # Créer une copie pour les modifications
    data_original_copy = data_original.copy()


    #*************************************************************************#
    #*************************************************************************#
    st.header("Pré-traitement des données")

    #----------------------------------#
    st.subheader("Création du train set et du test set")
    
    train_set, test_set = train_test_split(data_original, test_size=0.20, random_state=0)
    
    agree_show_train_set = st.checkbox('Afficher le train set',
                                  key="show_train_set")

    if agree_show_train_set:
        st.markdown("*Train set*")
        st.write(train_set)
        st.markdown("*Format des données*")
        st.write(train_set.shape)
     
           
    #----------------------------------#
    st.subheader("Feature selection simple 🗑️")
    
    var_to_delete_simple = st.multiselect("Sélectionnez les variables à supprimer", 
                                          data_original.columns,
                                          key="var_to_delete_simple ") 
    if var_to_delete_simple !=[]:
        data_original_copy = train_set.drop(var_to_delete_simple, axis=1)
        st.write(data_original_copy)

    
    #----------------------------------#
    st.subheader("Encodage des variables catégorielles")
        
    encoding_mth = st.selectbox("Sélectionnez la méthode d'encodage", 
                                ["Label Encoding", "One-Hot Encoding","Binary Encoding"],
                                key="encoding_mth") 
    
    
    if encoding_mth != None:
        # liste des colonnes type 'object'
        col_object = data_original_copy.select_dtypes(object).columns
            
        if encoding_mth=="Label Encoding":
            # créer l'encodeur
            label_encoder = LabelEncoder()
            for col in col_object:
                data_original_copy[col] = label_encoder.fit_transform(data_original_copy[col])
        
            # afficher le nouveau dataset
            st.markdown("*Train set après Label Encoding*")
            st.write(data_original_copy)
            
        elif encoding_mth=="One-Hot Encoding":
            # créer l'encodeur
            OH_encoder = OneHotEncoder(sparse=False)
                
            # appliquer l'encodeur : cela retourne un array
            OH_array = OH_encoder.fit_transform(data_original_copy[col_object])
            # transformer en dataframe + rajouter les noms de colonnes
            OH_df = pd.DataFrame(OH_array)
            # remettre les bons index
            OH_df.index = data_original_copy.index
            # supprimer les colonnes 'object' du dataset initial
            df_initial_num = data_original_copy.drop(col_object, axis=1)
            # concaténer les deux dataframe
            data_original_copy = pd.concat([df_initial_num,OH_df], axis=1)
            # afficher le nouveau dataset
            st.markdown("*Train set après One-Hot Encoding*")
            st.write(data_original_copy)
            
        elif encoding_mth=="Binary Encoding":
            # créer l'encodeur : on précise les colonnes à encoder
            binary_encoder = ce.BinaryEncoder(cols=col_object)
            # appliquer l'encodeur à nos données
            data_original_copy = binary_encoder.fit_transform(data_original_copy)
            # afficher le nouveau dataset
            st.markdown("*Train set après Binary Encoding*")
            st.write(data_original_copy)
            
    #----------------------------------#
    st.subheader("Données manquantes")
            
    st.write("Il n'y a aucune donnée manquante.")
    st.write(pd.DataFrame(data_original_copy.isna().sum()))
            
    #----------------------------------#
    st.subheader("Créer X_train et y_train")
    
    target = "DQR - Note de qualité de la donnée (1 excellente ; 5 très faible)"
    X_train = data_original_copy.drop(target,axis=1)
    y_train = data_original_copy[target]
            
    agree_show_X_train = st.checkbox('Afficher X_train',
                                     key="show_X_train")

    if agree_show_X_train:
        st.markdown("*X_train*")
        st.write(X_train)
        st.markdown("*Format des données*")
        st.write(X_train.shape)
        
        
        
        
        
        
     
    
    #*************************************************************************#
    #*************************************************************************#
    st.header("Premiers modèles")
    
    
    
    #-------------------------------------------------------------------------#
    st.markdown("""
                * On entraîne et on évalue les modèles dans leur version par défaut.
                * Outils d'évaluation :
                    * Métriques :
                        * MAE
                        * MSE
                        * RMSE
                        * $R^2$
                    * Learning Curve : [doc1](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html?highlight=learning#sklearn.model_selection.learning_curve), [doc2](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py)
                """)
    #-------------------------------------------------------------------------#
    
    
    #-------------------------------------------------------------------------#
    # on définit une fonction pour tracer les Learning Curve
    
    # paramètres :
    ## model : le modèle utilisé pour l'évaluation
    ## Xtrain, ytrain : données pour l'entraînement
    ## cv : nombre de cross-validation
    ## scoring : la métrique utilisé (donnée sous forme négative pour certains !)

    # output de 'learning curve' (il y en a 5 en tout, on n'en considère que 3 ici)
    ## N : array des tailles des échantillons utilisées pour l'entraînement
    ## train_score : array des scores pour chaque cross-validation, et chacun des échantillons d'entraînement 
    ## val_score : même chose pour les échantillons de validation

    # output de la fonction 'evaluation' : les learning curve (graphiques)
    
    def evaluation(model, Xtrain, ytrain, cv) :    
        
        
        fig, ax = plt.subplots(2,2)
        fig.tight_layout()
        
        #------------------------------------#
        # métrique 1
        # utilisation de la classe 'learning_curve' avec 'neg_mean_absolute_error'
        scoring = 'neg_mean_absolute_error'
        N, train_score, val_score = learning_curve(model,            
                                                   Xtrain, ytrain,
                                                   cv=cv,
                                                   scoring=scoring,
                                                   train_sizes=np.linspace(0.05,1,10))
        # afficher les train score
        ax[0][0].plot(N, -train_score.mean(axis=1), label='train score')
        # afficher les val score
        ax[0][0].plot(N, -val_score.mean(axis=1), label='validation score')
        # on rajoute une zone autour des courbes avec l'écart-type :
        # "courbe +/- écart-type"
        ax[0][0].fill_between(N, (-train_score).mean(axis=1) - (-train_score).std(axis=1),
                             (-train_score).mean(axis=1) + (-train_score).std(axis=1), alpha=0.1)
        ax[0][0].fill_between(N, (-val_score).mean(axis=1) - (-val_score).std(axis=1),
                             (-val_score).mean(axis=1) + (-val_score).std(axis=1), alpha=0.1)
        ax[0][0].legend()
        ax[0][0].set_title('MAE')
        #ax[0][0].xlabel("Taille de l'ensemble d'entraînement")
        
        #------------------------------------#
        # métrique 2
        # utilisation de la classe 'learning_curve' avec 'neg_mean_absolute_error'
        scoring = 'neg_mean_squared_error'
        N, train_score, val_score = learning_curve(model,            
                                                   Xtrain, ytrain,
                                                   cv=cv,
                                                   scoring=scoring,
                                                   train_sizes=np.linspace(0.05,1,10))
        # afficher les train score
        ax[0][1].plot(N, -train_score.mean(axis=1), label='train score')
        # afficher les val score
        ax[0][1].plot(N, -val_score.mean(axis=1), label='validation score')
        # on rajoute une zone autour des courbes avec l'écart-type :
        # "courbe +/- écart-type"
        ax[0][1].fill_between(N, (-train_score).mean(axis=1) - (-train_score).std(axis=1),
                             (-train_score).mean(axis=1) + (-train_score).std(axis=1), alpha=0.1)
        ax[0][1].fill_between(N, (-val_score).mean(axis=1) - (-val_score).std(axis=1),
                             (-val_score).mean(axis=1) + (-val_score).std(axis=1), alpha=0.1)
        ax[0][1].legend()
        ax[0][1].set_title('MSE')
        #ax[0][0].xlabel("Taille de l'ensemble d'entraînement")
       
        
        #------------------------------------#
        # métrique 3
        # utilisation de la classe 'learning_curve' avec 'neg_mean_absolute_error'
        scoring = 'neg_root_mean_squared_error'
        N, train_score, val_score = learning_curve(model,            
                                                   Xtrain, ytrain,
                                                   cv=cv,
                                                   scoring=scoring,
                                                   train_sizes=np.linspace(0.05,1,10))
        # afficher les train score
        ax[1][0].plot(N, -train_score.mean(axis=1), label='train score')
        # afficher les val score
        ax[1][0].plot(N, -val_score.mean(axis=1), label='validation score')
        # on rajoute une zone autour des courbes avec l'écart-type :
        # "courbe +/- écart-type"
        ax[1][0].fill_between(N, (-train_score).mean(axis=1) - (-train_score).std(axis=1),
                             (-train_score).mean(axis=1) + (-train_score).std(axis=1), alpha=0.1)
        ax[1][0].fill_between(N, (-val_score).mean(axis=1) - (-val_score).std(axis=1),
                             (-val_score).mean(axis=1) + (-val_score).std(axis=1), alpha=0.1)
        ax[1][0].legend()
        ax[1][0].set_title('RMSE')
        #ax[0][0].xlabel("Taille de l'ensemble d'entraînement")
       
        
        #------------------------------------#
        # métrique 4
        # utilisation de la classe 'learning_curve' avec 'neg_mean_absolute_error'
        scoring = "r2"
        N, train_score, val_score = learning_curve(model,            
                                                   Xtrain, ytrain,
                                                   cv=cv,
                                                   scoring=scoring,
                                                   train_sizes=np.linspace(0.05,1,10))
        # afficher les train score
        ax[1][1].plot(N, train_score.mean(axis=1), label='train score')
        # afficher les val score
        ax[1][1].plot(N, val_score.mean(axis=1), label='validation score')
        # on rajoute une zone autour des courbes avec l'écart-type :
        # "courbe +/- écart-type"
        ax[1][1].fill_between(N, (train_score).mean(axis=1) - (-train_score).std(axis=1),
                             (train_score).mean(axis=1) + (-train_score).std(axis=1), alpha=0.1)
        ax[1][1].fill_between(N, (val_score).mean(axis=1) - (-val_score).std(axis=1),
                             (val_score).mean(axis=1) + (-val_score).std(axis=1), alpha=0.1)
        ax[1][1].legend()
        ax[1][1].set_title('R2')
        #ax[0][0].xlabel("Taille de l'ensemble d'entraînement")
        
        
        st.pyplot(fig)    
    #-------------------------------------------------------------------------#

   
    
    #-------------------------------------------------------------------------#
    st.subheader("Modèles linéaires")
    
    models_linear_default = st.multiselect("Sélectionnez les modèles linéaires", 
                                          ['LinearRegression', 'Ridge', 'Lasso', 
                                           'ElasticNet', 'SGDRegressor', 
                                           "Régression Polynomiale"],
                                            key="models_linear_default") 
   
    #-----------------#
    # liste des noms des modèles
    model_name = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'SGDRegressor',
                  "Régression Polynomiale"]
    # liste des scoring
    #scoring_list = ['neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error', 'r2']
    # liste des noms des métriques
    # metric_name = ['MAE', 'MSE', 'RMSE', 'R2']
    # taille de la cross-validation
    c_v = 4
    #-----------------#
    
    
    
    if  models_linear_default != []:
        for model in models_linear_default:
            if model=='LinearRegression':
                # definir le modèle 
                model_lin_reg = LinearRegression()
                model_lin_reg.fit(X_train,y_train)
                st.write('LinearRegression')
                evaluation(model_lin_reg, 
                           X_train, y_train,
                           c_v)
                
            elif model=='Ridge':
                # definir le modèle 
                model_lin_ridge = Ridge()
                model_lin_ridge.fit(X_train,y_train)
                st.write('Ridge')
                evaluation(model_lin_ridge, 
                           X_train, y_train,
                           c_v)
                
    
            elif model=='Lasso':
                # definir le modèle 
                model_lin_lasso = Lasso()
                model_lin_lasso.fit(X_train,y_train)
                st.write('Lasso')
                evaluation(model_lin_lasso, 
                           X_train, y_train,
                           c_v)
                
            elif model=='ElasticNet':
                # definir le modèle 
                model_lin_ElasticNet = ElasticNet()
                model_lin_ElasticNet.fit(X_train,y_train)
                st.write('ElasticNet')
                evaluation(model_lin_ElasticNet, 
                           X_train, y_train,
                           c_v)
    
            elif model=='SGDRegressor':
                # definir le modèle 
                model_lin_SGDRegressor = SGDRegressor()
                model_lin_SGDRegressor.fit(X_train,y_train)
                st.write('SGDRegressor')
                evaluation(model_lin_SGDRegressor, 
                           X_train, y_train,
                           c_v)
    
            elif model=="Régression Polynomiale":
                # d'abord appliquer un Polynomial Featuring sur les données
                poly_features = PolynomialFeatures()
                X_train_poly = poly_features.fit_transform(X_train)
                # definir le modèle : on applique une régression linéaire, mais à 'X_train_poly'
                model_lin_reg = LinearRegression()
                model_lin_reg.fit(X_train_poly,y_train)
                st.write('LinearRegression')
                evaluation(model_lin_reg, 
                           X_train_poly, y_train,
                           c_v)
    
    
    #-------------------------------------------------------------------------#
  #  st.subheader("Machines à vecteurs de support")
    
    #-------------------------------------------------------------------------#
   # st.subheader("Méthodes des plus proches voisins")
    
    #-------------------------------------------------------------------------#
   # st.subheader("Arbres de décision")
    
    #-------------------------------------------------------------------------#
    #st.subheader("Méthodes ensemblistes")
    
    #-------------------------------------------------------------------------#
   # st.subheader("Réseaux de neurones")
    
    
    
    
    
#==============================   Page 4  ===================================#
#==================  Prédiction du DQR : Amélioration des modèles  ==========#
def page4():
    
    st.title("Prédiction du DQR : Amélioration des modèles")

    
#########################################################
if __name__=="__main__":
    main()









