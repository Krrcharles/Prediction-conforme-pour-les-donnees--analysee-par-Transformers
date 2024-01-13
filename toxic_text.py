import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = np.load('data/toxic-text/toxic-text-detoxify.npz')
preds = data['preds'] # Toxicity score in [0,1]
toxic = data['labels'] # Toxic (1) or not (0)

# Problem setup
alpha = 0.1 # 1-alpha is the desired type-1 error
n = 10000 # Use 200 calibration points

# Look at only the non-toxic data
nontoxic = toxic == 0
preds_nontoxic = preds[nontoxic]
preds_toxic = preds[np.invert(nontoxic)]

# Split nontoxic data into calibration and validation sets (save the shuffling)
idx = np.array([1] * n + [0] * (preds_nontoxic.shape[0]-n)) > 0
np.random.shuffle(idx)
cal_scores, val_scores = preds_nontoxic[idx], preds_nontoxic[np.invert(idx)]

# Use the outlier detection method to get a threshold on the toxicities
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n)
# Perform outlier detection on the ind and ood data
outlier_ind = val_scores > qhat # We want this to be no more than alpha on average
outlier_ood = preds_toxic > qhat # We want this to be as large as possible, but it doesn't have a guarantee

# Calculate type-1 and type-2 errors
type1 = outlier_ind.mean()
type2 = 1-outlier_ood.mean()
print(f"The type-1 error is {type1:.4f}, the type-2 error is {type2:.4f}, and the threshold is {qhat:.4f}.")

# Show some examples of unflagged and flagged text
content = pd.read_csv('generation-scripts/toxic_text_utils/test.csv')['content']
print("Unflagged text examples:")
print(list(np.random.choice(content[preds <= qhat],size=(5,))))
print("\n\nFlagged text examples:")
print(list(np.random.choice(content[preds > qhat],size=(5,))))