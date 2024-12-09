import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("notes.csv", sep=';')

X = data.set_index(data.columns[0]).transpose()

nomi = list(X.index)
nomv = list(X.columns)

french_grades = X['fran']
latin_grades = X['lati']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(french_grades, edgecolor='black', alpha=0.7)
ax1.set_title('Distribution des notes de Français')
ax1.set_xlabel('Notes')
ax1.set_ylabel('Frequency')

ax2.hist(latin_grades, edgecolor='black', alpha=0.7)
ax2.set_title('Distribution des notes de Latin')
ax2.set_xlabel('Notes')
ax2.set_ylabel('Frequency')

plt.tight_layout()

plt.show()
