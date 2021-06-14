# -*- coding: utf-8 -*-

##############################################################################
##############################################################################
#                                                                            #
#                         Importation des librairies                         #
#                                                                            #
##############################################################################
##############################################################################
import streamlit as st


# general
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, plot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
## fixer le theme
import plotly.io as pio
pio.templates.default = 'ggplot2'



# pour afficher les informations générales sur le dataset
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report

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
     #   "Prédire le DQR": page3,
    }

    st.sidebar.title('Navigation')
    page = st.sidebar.radio("", list(PAGES.keys()))
    PAGES[page]()
    
        
##############################################################################
# Fonctions correspondant aux différentes pages
##############################################################################


#==============================   Page 1  ===================================#
#===============================  Accueil  ==================================#
def page1():
    st.title('Agribalyse')
    st.write("##")
    st.write("""
             Bienvenue !  
             
             
             """)
             
    st.markdown(":tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:"+
                " :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:"+
                " :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:"+
                " :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:"+
                " :tractor:  :tractor:  :tractor:")         
    
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
                Dans la version actuelle, elle vous permet d'explorer les
                données à l'aide essentiellement de Plotly  \U0001f642 .
                """
                )
   
                
                
#==============================   Page 2  ====================================#
#========================  Exploration des données  ==========================#
def page2():
    
    
    st.sidebar.markdown("")
    
    #--Sélection du Thème des graphique----#
    theme_select = st.sidebar.selectbox("Choisissez le thème de vos graphiques pour la suite (il y a quelques conflits avec celui de Streamlit))" ,
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
    
    synthese_dataset = pd.read_csv("datasets/Agribalyse_Synthese.csv", header=0)
    st.write(synthese_dataset)
    
    #-------------------------------------------------------------------------#
    #-------------------------------------------------------------------------#
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
    
    #-------------------------------------------------------------------------#
    #-------------------------------------------------------------------------#
    st.header("2e niveau d'exploration")
    
    #-------------------------------------------------------------------------#
    st.subheader("Analyse univariée")
    
    #-------------------------------------------------------------------------#
    st.markdown("#### Variable continue")
    var_cont = st.selectbox("Sélectionnez une variable continue", 
                            synthese_dataset.select_dtypes(float).columns,
                            index=0)   # variable affichée par defaut
    
    number_bins = st.slider('Nombre de bins', min_value=4, max_value=500,
                            value=4)
    
    fig = px.histogram(synthese_dataset, x=var_cont, nbins=number_bins,
                       histnorm='probability',
                       marginal='box',
                       title='Histogramme de la variable {}'.format(var_cont))
    st.write(fig)
    
    #-------------------------------------------------------------------------#
    st.markdown("#### Variable catégorielle")
    var_cat = st.selectbox("Sélectionnez une variable catégorielle", 
                              synthese_dataset.select_dtypes(object).columns,
                              key="cat",
                              index=1)   # variable affichée par defaut
    
    fig = px.histogram(synthese_dataset, x=var_cat,
                       title='Distribution de la variable {}'.format(var_cat))
    fig.update_xaxes(categoryorder="total descending")
    st.write(fig)

    
    #-------------------------------------------------------------------------#
    st.subheader("Analyse bivariée")
    
    
    #-------------------------------------------------------------------------#
    st.markdown("#### Variables continues")
    
    # choisir le type de graphique
    type_cont = st.selectbox("Sélectionnez le type de graphique", 
                              ["Scatter Matrix", "Parallel Coordinates Plot"], 
                              index=0)    # variable affichée par defaut
    
    #-----------------------------------#
    if type_cont=="Scatter Matrix":
        var_conts_scatter_mat = st.multiselect("Sélectionnez les variables continues", 
                                               synthese_dataset.select_dtypes(float).columns,
                                               key="cont_scatter_matr")
        if var_conts_scatter_mat !=():
            fig = px.scatter_matrix(synthese_dataset,
                                    dimensions=var_conts_scatter_mat, 
                                    title='Scatter Matrix')
            # taille markers
            fig.update_traces(marker=dict(size=1))
            # enlever ou réduire les labels, valeurs, ticks qui rendent illisibles
            nb_col=len(var_conts_scatter_mat)  
            fig.update_layout({"xaxis" + str(i+1): dict(showticklabels=False, ticklen=0, titlefont=dict(size=(nb_col+3.8)/nb_col)) for i in range(nb_col)})
            fig.update_layout({"yaxis" + str(i+1): dict(showticklabels=False, ticklen=0, titlefont=dict(size=(nb_col+2.2)/nb_col)) for i in range(nb_col)})

            st.write(fig)
    
            st.markdown("""*Mmm ... Petit problème de labels à régler :confused:*""")
   
     #-----------------------------------#
    elif type_cont=="Parallel Coordinates Plot":
        var_conts_parallel = st.multiselect("Sélectionnez les variables continues", 
                                            synthese_dataset.select_dtypes(float).columns, 
                                            key="cont_parallel")
        if var_conts_parallel !=[]:
            fig = px.parallel_coordinates(synthese_dataset,
                                          dimensions=var_conts_parallel, 
                                          title='Parallel Coordinates Chart')
            st.write(fig)
   
    #-------------------------------------------------------------------------#
    st.markdown("#### Variables catégorielles")
    
    #st.markdown("*Malgré le message d'erreur quand le multiselect est vide, cela semble fonctionner ...  :confused:*")
    
    # choisir le type de graphique
    type_cat = st.selectbox("Sélectionner le type de graphique", 
                            ["Parallel categories plot", "Arbre"])
    
    #-----------------------------------#
    if type_cat=="Parallel categories plot":
        var_cats_parallel = st.multiselect("Sélectionnez les variables catégorielles", 
                                           synthese_dataset.select_dtypes(object).columns, 
                                           key="cats_parallel")
                                          
        if var_cats_parallel != []:
                fig = px.parallel_categories(synthese_dataset,
                                             dimensions=var_cats_parallel,
                                             title="Parallel categories plot")
                st.write(fig) 
                
    #-----------------------------------#
    elif type_cat=="Arbre":
        var_cats_tree = st.multiselect("Sélectionnez deux variables catégorielles", 
                                       synthese_dataset.select_dtypes(object).columns,
                                       key="cats_tree")
                                      
        if var_cats_tree != []:
            fig = px.treemap(synthese_dataset,
                             path=var_cats_tree,
                             title="Arbre")
            st.write(fig) 
            
            
 
 
    
  #-------------------------------------------------------------------------#
    st.markdown("#### Variables continues et catégorielles")
    
    #st.markdown(" *Malgré le message d'erreur quand le multiselect est vide, cela semble fonctionner ...  :confused: *")
    
    # choisir le type de graphique  
    type_cont_cat = st.selectbox("Sélectionnez le type de graphique", 
                                 ["Box plot", "Ridgeline", "Parallel categories plot"],
                                 key="cont_cat")
   
    
    #-----------------------------------#
    if type_cont_cat=="Box plot":
        var_cont1_cat2_box = st.multiselect("Sélectionnez votre variable continue, puis votre variable catégorielle", 
                                            synthese_dataset.columns,
                                            key="cont_1_cat2_box")
        if var_cont1_cat2_box != []:
            var_cont1_box=var_cont1_cat2_box[0]
            var_cat2_box=var_cont1_cat2_box[1]
            fig = px.box(synthese_dataset, 
                         x=var_cat2_box, 
                         y=var_cont1_box,
                         color=var_cat2_box,
                         title="{} en fonction de {}".format(var_cont1_box, var_cat2_box))
            fig.update_layout(boxgap=0, showlegend=False)
            fig.update_xaxes(tickangle=45)
            st.write(fig) 
    
    #-----------------------------------#
    if type_cont_cat=="Ridgeline":
        var_cont1_cat2_ridge = st.multiselect("Sélectionnez votre variable continue, puis votre variable catégorielle", 
                                            synthese_dataset.columns,
                                            key="cont_1_cat2_ridge")
        if var_cont1_cat2_ridge !=[]:
            var_cont1_ridge=var_cont1_cat2_ridge[0]
            var_cat2_ridge=var_cont1_cat2_ridge[1]
            fig = px.violin(synthese_dataset, 
                            x=var_cont1_ridge, 
                            y=var_cat2_ridge,
                            orientation='h', 
                            color=var_cat2_ridge,
                            title="{} en fonction de {}".format(var_cont1_ridge, var_cat2_ridge))
            fig.update_traces(side='positive', width=2)
            fig.update_layout(showlegend=False) 
            st.write(fig)  
            
            
    #-----------------------------------#
    if type_cont_cat=="Parallel categories plot":
        var_cat1_cont2_parallel = st.multiselect("Sélectionnez votre variable catégorielle, puis votre variable continue", 
                                                 synthese_dataset.columns,
                                                 key="cat_1_cont2_parallel")  
        
        if var_cat1_cont2_parallel !=[]:
            var_cat1_parallel=var_cat1_cont2_parallel[0]
            var_cont2_parallel=var_cat1_cont2_parallel[1]
            
            fig = px.parallel_categories(synthese_dataset, dimensions=[var_cat1_parallel], 
                                   color=var_cont2_parallel, 
                                   color_continuous_scale=px.colors.diverging.Tealrose, 
                                   color_continuous_midpoint=3)
            st.write(fig)
            
            
    

#########################################################
if __name__=="__main__":
    main()









