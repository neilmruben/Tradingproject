# Machine-Learning-trading


### Machine learning & trading : Dashboard interactif en Python avec Streamlit

Lien vers le dashboard : http://40.68.93.181:9090

Ce dashboard interactif permet de prédire le mouvement d’un actif (haussier / baissier) sélectionné au préalable. 
Il permet de choisir les paramètres des features utilisés ainsi que certains paramètres avant modèlisation comme la taille de l’échantillon de test 
et le nombre de kfold pour la cross-validation. Une fois la cross-validation effectuée on retiens les 3 meilleurs modèles de classification
pour notre prédiction à court-terme (ici une prédiction à intervalle de 2 minutes). 

Il permet une fois les modèles entrainés de télécharger ces modèles au format pickle pour les réutiliser (il est possible de charger 
directement sur le dashboard des fichiers pickle en enlevant les commentaires du code Python disponible sur github).
Les données sont normalisées avant la cross-validation avec la fonction MinMaxScaler de scikit-learn, on divsise chaque valeur par la différence entre la valeur maximale de notre colonne et la valeur minimale :

$$ x' = \frac{x - x_{min}}{x_{max} - x_{min}} $$

<img width="1470" alt="Capture d’écran 2024-02-18 à 14 24 32" src="https://github.com/neilmruben/Tradingproject/assets/81652761/fcd222c0-bb52-4939-8adf-e80e66d64d03">

<img width="732" alt="Capture d’écran 2024-02-18 à 14 29 53" src="https://github.com/neilmruben/Tradingproject/assets/81652761/d3bcf72e-614b-4f83-ab2d-be00c10aaef3">

<img width="732" alt="Capture d’écran 2024-02-18 à 14 30 18" src="https://github.com/neilmruben/Tradingproject/assets/81652761/57f7ca4c-6eee-43ce-ae53-f2695a26ad8b">

<img width="732" alt="Capture d’écran 2024-02-18 à 14 30 40" src="https://github.com/neilmruben/Tradingproject/assets/81652761/b2fb28b0-bcea-44a4-afbb-c1fd3e20e73e">




