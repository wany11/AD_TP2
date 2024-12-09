import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("notes.csv", sep=';')

X = data.set_index(data.columns[0]).transpose()

nomi = list(X.index)

math_scores = X['math']
science_scores = X['scie']
drawing_scores = X['d-m ']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for i in range(len(nomi)):
    ax1.scatter(math_scores[i], science_scores[i])
    ax1.text(math_scores[i], science_scores[i], nomi[i], fontsize=9, ha='right')
ax1.set_title('Mathematique vs Sciences')
ax1.set_xlabel('Mathematique')
ax1.set_ylabel('Sciences')

for i in range(len(nomi)):
    ax2.scatter(math_scores[i], drawing_scores[i])
    ax2.text(math_scores[i], drawing_scores[i], nomi[i], fontsize=9, ha='right')
ax2.set_title('Mathematique vs Dessin')
ax2.set_xlabel('Mathematique')
ax2.set_ylabel('Dessin')

plt.tight_layout()

plt.show()
