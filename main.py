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

# coefficient d'asymétrie
from scipy.stats import skew

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
    
    synthese_dataset = pd.read_csv("datasets/Agribalyse_Synthese.csv", header=0)
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
                       title='Histogramme de la variable {}'.format(var_cont))
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
                                    title='Scatter Matrix')
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
                                          title='Parallel Coordinates Chart')
            st.write(fig)
           
    
    #-----------------------------------#
    elif type_cont=="(option) Parallel Coordinates Plot":
        var_conts_parallel = st.multiselect("Sélectionnez des variables continues", 
                                            synthese_dataset.select_dtypes(float).columns, 
                                            key="cont_parallel"
                                            )   
        if var_conts_parallel !=[]:
            fig = px.parallel_coordinates(synthese_dataset,
                                          dimensions=var_conts_parallel, 
                                          title='Parallel Coordinates Chart')
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
                         title="Box plot")
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
                            title="Ridgeline".format(target, cat2_ridge))
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
                                         color_continuous_scale=px.colors.diverging.Tealrose, 
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
                             color_continuous_scale=px.colors.diverging.Tealrose,
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
                                title="Scatter matrix des variables numériques, sans la variable cible")
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
                                     title='Scatter Matrix')
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
                                          title='Parallel Coordinates Chart')
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
                               color_discrete_sequence=px.colors.qualitative.Set3)            
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
                         title="{} en fonction de {}".format(var_cont1_box, var_cat2_box))
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
                            title="{} en fonction de {}".format(var_cont1_ridge, var_cat2_ridge))
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
                                         color_continuous_scale=px.colors.diverging.Tealrose, 
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
                             title="Treemap (proportions et couleurs avec la variable continue)")
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
    
  
    
#########################################################
if __name__=="__main__":
    main()









