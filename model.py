import pickle
import numpy as np
import pandas as pd

import urllib
filename = "data_.xlsx"
file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
urllib.request.urlretrieve(file_url, filename)

data_xls = pd.read_excel('data_.xlsx')
data_xls.to_csv('data.csv', index = None)
data = pd.read_csv('data.csv', index_col = False)

# Transaction date is in float let's make it integer

data["X1 transaction date"]= data["X1 transaction date"].astype(int)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = data.iloc[:,:7]
X = X.drop(["No"],axis=1)
y = data.iloc[:,7]

# Modelling
rr = RandomForestRegressor()
rr.fit(X,y)

# Saving model with pickle
pickle.dump(rr, open('pickle_file.pkl','wb'))

model = pickle.load(open('pickle_file.pkl','rb'))
print(model.predict([[2013,30,300,3,30,121]]))
