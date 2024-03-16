# Projet stat 2A Prédiction conforme pour les données textuelles analysées par Transformers

Voici les principales composantes du projet comme je le vois et que je souhaiterai voir :

0) Que sont nos données ?

--

1) Traitement des données

Tables de grande taille contenants notamment les rapports médicaux, composantes principales de notre projet.
Pour pouvoir les utiliser dans le cadre de notre projet statistique, il est nécessaire d'appliquer du NLP (Natural langage processing).
Les textes sont découpés en petit morceaux puis transformés en embedding par un modèle de type Transformers par ClinicalBert.
Ces embeddings étant de très grande taille, il est nécessaire de réduire leur dimension pour pouvoir les utiliser.

-> deux choix possibles : Projection gaussienne aléatoire VS ACP

Suite à cela, nous allons utilisé les log signatures de ces embeddings car plus riche en information.
Nous voilà avec un jeu de donné contenant plus de 5100 colonnes.

2) Apprentissage

Plusieurs choix de prédiction s'est offert à nous. Nous avons dans un premier temps voulut prédire la mort ou non d'un patient.
C'est une variale binaire présente déjà dans notre jeu de données. Il s'agit donc d'un problème de classification à deux classes.

-> deux algo de ML utilisés : Régression logistique VS Forêts aléatoires

Le cadre d'un prédiction entre deux classes n'étant pas "l'idéal" pour appliquer le but de notre projet statistique, il nous ait apparut évident
qu'il fallait changer de perspective. Après quelques discussions et du fait d'un jeu de donnés désquilibré entre les patients mort ou non durant leur
séjour à l'hopital, nous avons décidé dans un premier temps de nous focaliser sur la prédiction de la durée de séjour de chacun des patients (en heures).
Ceci à l'avantage d'être renseigné pour chacun des patients et reste d'un point de vue théorique pertinent et intéressant.
Il s'agit d'un problème de régression car variable continue.

-> deux algo de ML utilisés : Forêts aléatoires VS Régression quantiles via Forêts aléatoires

Le premier régresseur prédit la moyenne conditionnelle de Yi sachant Xi = x tandis que le deuxième prédit des quantiles conditionnellement à la distribution.
Cette deuxième option est utilisée car elle a des meilleurs propriétés pour appliquer la prédiction conforme que nous allons détaillés après.

3) Prédiction conforme

---

4) Comparaisons / Résultats intéressants / Bilan

---

5) Difficultés rencontrées

---
