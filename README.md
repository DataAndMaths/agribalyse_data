# Agribalyse-Synthèse Application 

----------------------
:tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  :tractor:  

:apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  :apple:  

:croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  :croissant:  

:truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  :truck:  
             
:fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  :fork_and_knife:  

---------------

## Bienvenue !

Voici une réutilisation des [données](https://datascience.etalab.studio/dgml/c763b24a-a0fe-4e77-9586-3d5453c631cd) d'Agribalyse :

[Lien vers l'application](https://share.streamlit.io/dataandmaths/agribalyse_data/main/main.py) (en construction :construction:)

NB : Il y a une mise à jour très récente de Streamlit, du coup, c'est lent, ça bug, ... 😕

------------------------------------

## Cadre

### Qu'est-ce qu'Agribalyse ? 

[AGRIBALYSE](https://doc.agribalyse.fr/documentation/) est une base de données de référence des indicateurs d’impacts environnementaux des produits agricoles produits en France et des produits alimentaires consommés en France. 

### Que trouve-t-on dans ces données ?

Elles recensent des caractéristiques de plusieurs aliments ainsi que les émissions de polluants qui leur sont associés.

### Que pouvez-vous faire avec cette petite application ? 
Dans la version actuelle, elle vous permet :
* d'explorer les données à l'aide essentiellement de Plotly 🙂 (graphiques interactifs)
* de construire un tout premier modèle pour prédire le DQR en fonction de différents indicateurs.

### Qu'est-ce que le DQR

Une note de qualité - le **Data Quality Ratio (DQR)** - de 1, très bon, à 5, très mauvais - est associée à chaque produit agricole et alimentaire pour 
lequel Agribalyse fournit des inventaires de cycle de vie et des indicateurs d’impacts. La Commission Européenne recommande de la prudence dans l’utilisation des données avec des DQR supérieurs à 3. 
Dans la base de données AGRIBALYSE, 67 % des données ont un DQR jugé bon ou très bon (1 à 3).
              
   
 --------------------------  
   
## Mon analyse des données
Le fichier étant volumineux (graphiques Plotly), j'ai dû découper le fichier en plusieurs parties. 
* [Partie 1](https://nbviewer.jupyter.org/github/DataAndMaths/agribalyse_data/blob/main/mes_notesbooks/Agribalyse_Synthese_EDA_%281%29_partie1.ipynb)
* [Partie 2A](https://nbviewer.jupyter.org/github/DataAndMaths/agribalyse_data/blob/main/mes_notesbooks/Agribalyse_Synthese_EDA_%281%29_Partie2A.ipynb)
* [Partie 2B](https://nbviewer.jupyter.org/github/DataAndMaths/agribalyse_data/blob/main/mes_notesbooks/Agribalyse_Synthese_EDA_%281%29_Partie2B.ipynb)
* [Partie 2C](https://nbviewer.jupyter.org/github/DataAndMaths/agribalyse_data/blob/main/mes_notesbooks/Agribalyse_Synthese_EDA_%281%29_Partie2C.ipynb)
* [Partie 3](https://nbviewer.jupyter.org/github/DataAndMaths/agribalyse_data/blob/main/mes_notesbooks/Agribalyse_Synthese_EDA_%281%29_Partie3.ipynb)
* [Partie 4](https://nbviewer.jupyter.org/github/DataAndMaths/agribalyse_data/blob/main/mes_notesbooks/Agribalyse_Synthese_EDA_%281%29_Partie4.ipynb)
