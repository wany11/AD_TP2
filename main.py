from scipy.io import loadmat
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from affichage_acp import my_biplot

data = pd.read_csv("notes.csv", sep=';')

names = data.set_index(data.columns[0]).transpose()

nomi = list(names.index)
nomv = list(names.columns)

X = data.set_index(data.columns[0]).transpose().values

p = 5
acp = PCA(n_components=p)

cc = acp.fit_transform(X)

acp2 = PCA(n_components='mle', whiten=True)

data2proj = acp2.fit_transform(X)

explained_variance = np.cumsum(acp.explained_variance_ratio_)


correlation = acp.components_ * np.sqrt(acp.explained_variance_ / X.var(axis=0) )

biplot = my_biplot(score=cc[:, 0:2] ,coeff=np.transpose(acp.components_[0:2, :]),coeff_labels=nomv, score_labels=nomi, nomx="PC1", nomy="PC2")

print("Principal components:\n", acp.components_)
print("Explained variance:\n", acp.explained_variance_)

plt.figure(figsize=(8, 6))
plt.scatter(cc[:, 0], cc[:, 1], alpha=0.7, c='blue', edgecolor='k', s=50)
for i, label in enumerate(nomi):
    plt.annotate(label, (cc[i, 0], cc[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center')

plt.title('PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.axhline(0, color='grey', lw=0.8)
plt.axvline(0, color='grey', lw=0.8)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1),
         explained_variance, marker='o')

plt.title('Cumulative Explained Variance by Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.xticks(range(1, len(explained_variance) + 1))
plt.show()


# Représentation E1 ∪ E2
plt.figure(figsize=(8, 6))
plt.scatter(data2proj[:, 0], data2proj[:, 1], color='blue', edgecolor='k', s=50)
for i, label in enumerate(nomi):
    plt.annotate(label, (data2proj[i, 0], data2proj[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.title('Projection on E1 ∪ E2')
plt.xlabel('Principal Component 1 (E1)')
plt.ylabel('Principal Component 2 (E2)')
plt.axhline(0, color='grey', lw=0.8)
plt.axvline(0, color='grey', lw=0.8)
plt.grid(True)

# Représentation E1 ∪ E3
plt.figure(figsize=(8, 6))
plt.scatter(data2proj[:, 0], data2proj[:, 2], color='red', edgecolor='k', s=50)
for i, label in enumerate(nomi):
    plt.annotate(label, (data2proj[i, 0], data2proj[i, 2]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.title('Projection on E1 ∪ E3')
plt.xlabel('Principal Component 1 (E1)')
plt.ylabel('Principal Component 3 (E3)')
plt.axhline(0, color='grey', lw=0.8)
plt.axvline(0, color='grey', lw=0.8)
plt.grid(True)
plt.show()

# Corrélation
print("Correlation:\n", correlation)