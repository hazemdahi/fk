import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

path2 = 'E:\\python\\1 project\\last .txt'
df = pd.read_csv(path2, header=None, names=['so2', 'no2','o3','MP10'])

# take a look at the dataset
#df.head()

#use required features
cdf = df[['so2','no2','o3','MP10']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]


regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))

'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''
