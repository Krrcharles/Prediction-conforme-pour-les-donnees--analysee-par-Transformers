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
C'est une variable binaire présente déjà dans notre jeu de données. Il s'agit donc d'un problème de classification à deux classes.

-> deux algo de ML utilisés : Régression logistique VS Forêts aléatoires

Le cadre d'un prédiction entre deux classes n'étant pas "l'idéal" pour appliquer le but de notre projet statistique, il nous ait apparut évident
qu'il fallait changer de perspective. Après quelques discussions et du fait d'un jeu de donnés déséquilibré entre les patients mort ou non durant leur
séjour à l'hôpital, nous avons décidé dans un premier temps de nous focaliser sur la prédiction de la durée de séjour de chacun des patients (en heures).
Ceci à l'avantage d'être renseigné pour chacun des patients et reste d'un point de vue théorique pertinent et intéressant.
Il s'agit d'un problème de régression car variable continue.

-> deux algo de ML utilisés : Forêts aléatoires VS Régression quantiles via Forêts aléatoires

Le premier régresseur prédit la moyenne conditionnelle de Yi sachant Xi = x tandis que le deuxième prédit des quantiles conditionnellement à la distribution.
Cette deuxième option est utilisée car elle a des meilleurs propriétés pour appliquer la prédiction conforme que nous allons détaillés après.

3) Prédiction conforme

L'intérêt de la prédiction conforme est qu'au lieu de prédire seulement une valeur par nos algorithmes de machine learning,
nous prédisons un set de valeurs possibles qui a une propriété de couverture intéressante asymptotiquement. En effet, on peut 
pour un risque donné alpha, choisir une couverture pour laquelle  approximativement 1 - alpha % des vrais valeurs sont contenues dans le set. Un set peut être un intervalle de valeurs pour des variables continues ou bien contenir des labels pour des problèmes de classification. La prédiction conforme comporte diverses avantages qui sont les suivants :
- Elle est indépendante de la précision du prédicteur (Bien que cela influe la taille du set) et dépend du choix de fonction de score.
- Aucune hypothèse n'est formulé sur la distribution des données. Uniquement une hypothèse d'échangeabilité des données.
- Elle est plus ou moins adaptative au sens où la taille du set nous donne une information sur la précision de notre prédiction.

Il existe différentes méthodes de prédiction conforme selon le problème de ML. 
Dans le cas de notre problème de classification à deux classes, nous avons appliqué des méthodes fournissant à priori des sets de faible taille ou plus adaptatifs. Il existe également plusieurs branches de ces méthodes en choisissant de ré-entraîner notre modèle pour chacune des valeurs possibles de notre variable de prédiction (appelé Full conformal) ou bien en décidant d'utiliser des données pour calibrer (appelé Split/induction conformal). La méthode Split est statistiquement moins précise du fait d'une perte en donnée d'entraînement. Toutefois, les temps de calcul pour la méthode Full conformal sont si élevé qu'il parait judicieux de se réserver un ensemble de donné pour calibrer la prédiction conforme.
Néanmoins, comme il s'agit d'un problème de classification a seulement deux classes, l'intérêt de la prédiction conforme reste assez limité. Les sets prenant la plupart du temps deux valeurs et rarement une seule (les sets vides qui sont théoriquement possible n'arrive pas dans le cas à deux classes).
C'est pourquoi nous avons décidé de faire un virage vers un problème de régression. Pour notre forêt aléatoire appliqué à une variable continue, nous avons appliqué une méthode de type split conformal et CV + à K (=5) folder. La méthode CV+ repose sur un entraînement du régresseur sur des données d'entraînement divisé K fois et des calculs de score selon une formule définie. Nous avons également pensé à utiliser la méthode Jacknife + qui a également de bonnes propriétés de couverture. Mais de même, il était nécessaire d'entraîner n fois notre modèle où n est la taille de notre échantillon d'entraînement ce qui n'était pas raisonnable techniquement. 
De plus, pour fournir un intervalle de prédiction, il parait au premier abord pertinent d'utiliser un régresseur de type quantile et de prédire les quantiles d'ordre alpha/2 et d'ordre 1-alpha/2. Toutefois, rien ne garantis que dans alpha % des cas la vrai valeur est contenu dans l'intervalle formé par les quantiles. C'est pourquoi il existe une méthode de type conformal quantile regression que nous avons appliqué qui repose sur une petite correction des intervalles.

4) Comparaisons / Résultats intéressants / Bilan

- Discuter de la qualité de nos prédicteurs :

----

- Discuter de la qualité des intervalles de prédictions :

Pour pouvoir différencié laquelle est la meilleure méthode de prédiction conforme appliqué à nos jeux de données, il parait pertinent d'observer nos résultats à travers différents spectres qui sont le respect de la couverture (proche de 1-alpha %), la taille du set (un set plus petit indique que notre prédicteur est plus sûr de sa prédiction est donc une plus faible incertitude) et dans un second temps le temps nécessaire pour appliquer ces méthodes.

On peut déjà faire remarquer que la taille du set de prédiction pour la méthode split est constante. Elle est juste centré sur valeur prédite. Cette taille d'intervalle étant calculé à travers le jeu de calibration. Toutefois, comme nous savons que la variable prédite est positive ou nulle, les sets sont tronqués prenant en compte cette spécificité. Dans le cas de la régressions quantile, les tailles sont assez variables qui peuvent être assez grands ou faibles. On peut différencier les deux méthodes en regardant la taille moyenne des sets sur le jeu de test. Or la spécificité de la regression quantile fait qu'il est bien plus intéressant du fait d'une adaptation aux données rencontrés et la difficulté induite par la prédiction ou non. C'est pourquoi cette méthode apparaît plus pertinente que l'autre. 

5) Difficultés rencontrées

La principale difficulté rencontré est le temps de calcul nécessaire pour entraîner les modèles que ca soit au niveau du Bert, des régresseurs ou bien des méthodes de prédiction. Les temps nécessaires de calcul dépassant les dizaines d'heures pour certains, ont été un frein pour le projet. Le projet n'a pas été commencé de zéro par nous. Notre tuteur nous a fourni une bonne base de code allant jusqu'à la création des prédicteurs. Cette base indispensable pour qu'on ait pu aller à ce niveau en fin de projet, a nécessité un certain temps pour réussir à le faire tourner sur nos machines, à comprendre les aspects techniques (NLP, signature, projection gaussienne aléatoire, ...).
De plus, le choix du passage de la variable binaire à la variable continue à nécessité diverses discussions sur le choix de la variable à prédire et des difficultés sous-jacentes (problème de censure à droite dans le cas d'une prédiction de la durée de vie).