import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

arr = pd.DataFrame({'val':[1,2,3,1,1,1,1,1,2,3,3,4,4,4,5]})

sns.countplot(data=arr, x='val')

