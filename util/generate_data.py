#Filename:	generate_data.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 16 Mar 2021 09:31:25  WIB

import numpy as np
import pandas as pd

num = 1000

mean = [8000, 20]
std = [[2000, 0], [0, 2]]

x1, x2 = np.random.multivariate_normal(mean, std, 1000).T
y1 = np.ones(1000)

positive = np.stack((x1, x2, y1)).T

mean1 = [3000, 15]
std1 = [[2000, 0], [0, 2]]

x11, x22 = np.random.multivariate_normal(mean1, std1, 1000).T
y0 = np.zeros(1000)

negative = np.stack((x11, x22, y0)).T
all_data = np.concatenate((positive, negative))
index = np.random.permutation(len(all_data))
all_data = all_data[index]

csv_file = pd.DataFrame.from_records(columns = ['salary', 'education', 'label'], data = all_data)
csv_file.to_csv("Synthetic_data.csv", index = False)
