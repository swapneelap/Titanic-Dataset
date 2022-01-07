import pandas as pd

data_source = '/home/swapneel/Projects/Titanic Dataset/data/'
data = pd.read_csv(data_source+'train.csv')
print(data.info())
print(data['Cabin'].value_counts())
print(data['Pclass'].value_counts())
# data.hist(figsize=(20, 15))
# attributes = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
# pd.plotting.scatter_matrix(data[attributes], figsize=(14, 10))
