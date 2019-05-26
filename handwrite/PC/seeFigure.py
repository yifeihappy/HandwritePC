import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# file_path = "./runtime.txt"
file_path = "./initial.txt"

# dataFrame = pickle.load(file_path, sep=",")
dataFrame = pd.read_table(file_path, sep=',')
data = dataFrame.values.copy()
magnetic = data[data[:, 0] == 2, :]
magnetic_start = magnetic[0, :]
magnetic = magnetic - magnetic_start
t = magnetic[:, 1]
fig, ax = plt.subplots(1, 1)
ax.plot(t, magnetic[:, 2], 'r', label='x')
ax.plot(t, magnetic[:, 3], 'g', label='y')
ax.plot(t, magnetic[:, 4], 'b', label='z')
handles, labels = ax.get_legend_handles_labels()
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
ax.legend(handles, labels, prop=font1)
plt.tick_params(labelsize=20)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
fig.show()
