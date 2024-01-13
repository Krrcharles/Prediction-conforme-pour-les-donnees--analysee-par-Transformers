import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

data = np.load('data/imagenet/imagenet-resnet152.npz')
example_paths = os.listdir('data/imagenet/examples')
smx = data['smx']
labels = data['labels'].astype(int)

# Problem setup
n = 1000 # number of calibration points
alpha = 0.1 # 1-alpha is the desired coverage

# Split the softmax scores into calibration and validation sets (save the shuffling)
idx = np.array([1] * n + [0] * (smx.shape[0]-n)) > 0
np.random.shuffle(idx)
cal_smx, val_smx = smx[idx,:], smx[~idx,:]
cal_labels, val_labels = labels[idx], labels[~idx]

# 1: get conformal scores. n = calib_Y.shape[0]
cal_scores = 1-cal_smx[np.arange(n),cal_labels]
# 2: get adjusted quantile
q_level = np.ceil((n+1)*(1-alpha))/n
qhat = np.quantile(cal_scores, q_level, interpolation='higher')
prediction_sets = val_smx >= (1-qhat) # 3: form prediction sets

# Calculate empirical coverage
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