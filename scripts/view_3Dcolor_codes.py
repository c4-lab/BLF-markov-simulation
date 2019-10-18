"""This script is for viewing 3-d plot for rgb color codes."""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns


# read color codes file in csv 
df = pd.read_csv('color_codesnode2Vec.csv')


fig = plt.figure()
ax = plt.axes(projection='3d')

cmap = sns.cubehelix_palette(as_cmap=True)

ax.scatter3D(df.b, df.g, df.r, cmap=cmap)
plt.title('Node2Vec')    
ax.set_xlabel('Blue')
ax.set_ylabel('Green')
ax.set_zlabel('Red')

plt.show()
