import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

data = np.load('data/imagenet/imagenet-resnet152.npz')
example_paths = os.listdir('data/imagenet/examples')
smx = data['smx']
labels = data['labels'].astype(int)

# Il y a 50 000 images dans le jeu de données data
# Chaque image a un numéro (example_indexes) et un label associé
# Il y a 1 000 labels différents
# Un élement de la table smx (une image) est une liste de 1 000 valeurs (softmax scores) obtenues à partir
# d'un prédicteur (supposément réseaux de neurones).

print(list(data.keys()))
print(data['smx'][:3])
print("//")
print(data['labels'][:3])
print("//")
print(data['example_indexes'][:3])
print("//")

# Problem setup
n = 1000 # number of calibration points
alpha = 0.1 # 1-alpha is the desired coverage

# Split the softmax scores into calibration and validation sets (save the shuffling)
idx = np.array([1] * n + [0] * (smx.shape[0]-n)) > 0
np.random.shuffle(idx)
# ix est un vecteur de taille 50 000 (nb d'images) qui contient n True et 50 000 - n False disposé aléatoirement
cal_smx, val_smx = smx[idx,:], smx[~idx,:]
cal_labels, val_labels = labels[idx], labels[~idx]

# 1: get conformal scores. n = calib_Y.shape[0]
cal_scores = 1-cal_smx[np.arange(n),cal_labels]
# Pour chacunes des images du set de calibration, on récupère 1-softmax associé au vrai label (liste de n éléments)

# 2: get adjusted quantile
q_level = np.ceil((n+1)*(1-alpha))/n # 0.9 environ 
qhat = np.quantile(cal_scores, q_level, interpolation='higher') # valeur du 9 ème décile environ

# 3: form prediction sets
prediction_sets = val_smx >= (1-qhat) 
# Pour chaque image du set de validation, on garde les labels dont le softmax dépasse le seuil (on remplace les valeurs par True/False)

# Calculate empirical coverage
# On regarde en moyenne si le set de prédiction contient bien le vrai label
empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
print(f"The empirical coverage is: {empirical_coverage}")

# Show some examples
with open('data/imagenet/human_readable_labels.json') as f:
    label_strings = np.array(json.load(f))

example_paths =os.listdir('data/imagenet/examples')
for i in range(10):
    rand_path = np.random.choice(example_paths)
    img = imread('data/imagenet/examples/' + rand_path )
    img_index = int(rand_path.split('.')[0])
    prediction_set = smx[img_index] > 1-qhat
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    print(f"The prediction set is: {list(label_strings[prediction_set])}")

# Les sets de prédiction vide ou de grande taille soulignent une incertitude du prédicteur
# Tandis qu'un set d'un seul élèment témoigne d'une plus grande confiance dans la prédiction