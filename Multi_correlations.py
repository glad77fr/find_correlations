import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

wine = datasets.load_wine()

wine = pd.DataFrame(wine['data'], columns=wine['feature_names'])
#wine['target'] = pd.Series(wine.target)
corr = wine.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin='-1', vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(wine.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(wine.columns)
ax.set_yticklabels(wine.columns)
plt.show()

