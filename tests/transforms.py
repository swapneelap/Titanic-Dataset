import pandas as pd
import sklearn.base
import numpy as np

data_source = '/home/swapneel/Projects/Titanic Dataset/data/'
data = pd.read_csv(data_source+'train.csv')
print(data.columns)


class ReplaceStr(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Custom transformer to replace strings with a string in a column"""

    def __init__(self, column_ix, replacement_string_list, if_nan=np.nan):
        self.column_ix = column_ix  # Column index
        self.replacement_string_list = replacement_string_list
        self.if_nan = if_nan

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for index in range(X[:, self.column_ix].shape[0]):
            if pd.isnull(X[:, self.column_ix][index]):
                X[:, self.column_ix][index] = self.if_nan
            for sub_str in self.replacement_string_list:
                if sub_str in X[:, self.column_ix][index]:
                    X[:, self.column_ix][index] = sub_str

        return X


titles = ['Mrs.', 'Mr.', 'Master.', 'Miss.', 'Major.', 'Rev.', 'Dr.', 'Ms.',
          'Mlle.', 'Col.', 'Capt.', 'Mme.', 'Countess.', 'Don.', 'Jonkheer.',
          'Sir.', 'Lady.']

data_array_y = data['Survived'].to_numpy()
data.drop('Survived', axis=1, inplace=True)
print(data.columns)
data_array_X = data.to_numpy()
# print(type(data_array[0][10]))  # The Name column index is 3

transformer_1 = ReplaceStr(2, titles)
data_array_X = transformer_1.fit_transform(data_array_X)
print(data_array_X[:, 2])

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
transformer_2 = ReplaceStr(9, cabin_list, 'U')
data_array_X = transformer_2.fit_transform(data_array_X)
print(data_array_X[:, 9])
# test_array = data_array_X[np.where(data_array_X[:, 2] == 'Mr')][:, 4]
# print(np.mean(test_array[~ pd.isnull(test_array)]))


class AgeImputer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Class to fill empty age values of passangers

    It takes into account the Name (Title) of the passangers
    as well.
    """

    def __init__(self):
        self.age_ix = 4
        self.name_ix = 2
        self.title_list = ['Mrs.', 'Mr.', 'Master.', 'Miss.', 'Major.', 'Rev.',
                           'Dr.', 'Ms.', 'Mlle.', 'Col.', 'Capt.', 'Mme.',
                           'Countess.', 'Don.', 'Jonkheer.', 'Sir.', 'Lady.']
        self.mean_dict_ = dict()

    def fit(self, X, y=None):
        for title in self.title_list:
            age_array_raw = X[np.where(X[:, self.name_ix] == title)][:, self.age_ix]
            self.mean_dict_[title] = np.mean(age_array_raw[~ pd.isnull(age_array_raw)])
        return self

    def transform(self, X, y=None):
        for index in np.where(pd.isnull(X[:, self.age_ix]))[0]:
            title = X[index][self.name_ix]
            X[index][self.age_ix] = self.mean_dict_[title]

        return X


transformer_3 = AgeImputer()
transformer_3.fit(data_array_X)
print(transformer_3.mean_dict_['Mr.'])
data_array_X = transformer_3.transform(data_array_X)
