import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("notes.csv", sep=';')

X = data.set_index(data.columns[0]).transpose()

nomi = list(X.index)
nomv = list(X.columns)


french_grades = X['fran']

plt.hist(french_grades, edgecolor='black')
plt.title('Distribution des notes de Français')
plt.xlabel('Notes')
plt.ylabel('Freq')
plt.show()
