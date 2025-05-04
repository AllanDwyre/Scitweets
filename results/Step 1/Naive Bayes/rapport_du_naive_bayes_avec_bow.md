# Rapport du Naive Bayes avec BoW
**Date**: 2025-05-03_20-56-22

## Description
Ce graphique montre le résultats hyperparamètrique, avec GridCV

param_grid = {
	'preprocessor__text__max_features': [5000, 10000, 20000],
	'preprocessor__text__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
	'preprocessor__text__min_df': [1, 2, 5],
	
	'classifier__alpha': [0.01, 0.1, 0.5, 1.0],
}

Le problème que je me rends compte c'est que le résultat est moins bien que le modele de base (il y a un terme pour le modele de base)

Donc le prochain choix est de réduire les paramètres a tester
## Visualisation
![Rapport du Naive Bayes avec BoW](../../static/images/rapport_du_naive_bayes_avec_bow_plot.png)
