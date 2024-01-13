import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.optimize import brentq

data = np.load('data/coco/coco-tresnetxl.npz')
example_paths = os.listdir('data/coco/examples')

sgmd = data['sgmd'] # sigmoid scores
labels = data['labels']
example_indexes = data['example_indexes']

# Problem setup
n=1000 # number of calibration points
alpha = 0.1 # 1-alpha is the desired false negative rate

def false_negative_rate(prediction_set, gt_labels):
    return 1-((prediction_set * gt_labels).sum(axis=1)/gt_labels.sum(axis=1)).mean()

# Split the softmax scores into calibration and validation sets (save the shuffling)
idx = np.array([1] * n + [0] * (sgmd.shape[0]-n)) > 0
np.random.shuffle(idx)
cal_sgmd, val_sgmd = sgmd[idx,:], sgmd[~idx,:]
cal_labels, val_labels = labels[idx], labels[~idx]

# Run the conformal risk control procedure
def lamhat_threshold(lam): return false_negative_rate(cal_sgmd>=lam, cal_labels) - ((n+1)/n*alpha - 1/(n+1))
lamhat = brentq(lamhat_threshold, 0, 1)
prediction_sets = val_sgmd >= lamhat

# Calculate empirical FNR
print(f"The empirical FNR is: {false_negative_rate(prediction_sets, val_labels)} and the threshold value is: {lamhat}")

# Show some examples
label_strings = np.load('data/coco/human_readable_labels.npy')

example_paths =os.listdir('data/coco/examples')
for i in range(10):
    rand_path = np.random.choice(example_paths)
    img = imread('data/coco/examples/' + rand_path )
    img_index = int(rand_path.split('.')[0])
    prediction_set = sgmd[img_index] > 1-lamhat
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    print(f"The prediction set is: {list(label_strings[prediction_set])}")