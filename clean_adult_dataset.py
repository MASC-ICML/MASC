import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    dataframe = pd.read_csv(full_path, header=None, na_values='?')
    # drop rows with missing
    dataframe = dataframe.dropna()
    # split into inputs and outputs
    last_ix = len(dataframe.columns) - 1
    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
    # select categorical and numerical features
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    y = np.where(y == '>50K', 1, -1)
    
    return X, y, cat_ix, num_ix

# define the location of the dataset
full_path = 'datasets/adult/adult-all.csv'
# load the dataset
df, Y, cat_ix, num_ix = load_dataset(full_path)
# summarize the loaded dataset
# print(df.shape, Y.shape, Counter(Y))

# Explanation of column meanings : https://cseweb.ucsd.edu//classes/sp15/cse190-c/reports/sp15/048.pdf

column_names = {
 0 : "age", # Integer
 1 : "work_class", # Categorical, 8
 2 : "final_weight", # Integer
 3 : "education", # Categorical, 16
 4 : "education_num", # Integer, fully correlated with "education"
 5 : "marital_status", # Categorical, 7
 6 : "occupation", # Categorical, 14
 7 : "relationship", # Categorical, 6
 8 : "race", # Categorical, 5, GROUP CATEGORY
 9 : "sex", # Categorical, 2
 10 : "capitol_gain", # Integer
 11 : "capitol_loss", # Integer
 12 : "hours_per_week", # Continuous
 13 : "native_country", # Categorical, 41
}

df.rename(columns=column_names, inplace=True)
df.drop("education", axis=1, inplace=True)
df.drop("native_country", axis=1, inplace=True)

# Clean up occupation column by removing any leading/trailing spaces
df['occupation'] = df['occupation'].str.strip()
# Replace occupation categories with new categories
df['new_occupation'] = df['occupation'].replace({
    'Prof-specialty': 'Professional_Managerial',
    'Craft-repair': 'Skilled_Technical',
    'Exec-managerial': 'Professional_Managerial',
    'Adm-clerical': 'Sales_Administrative',
    'Sales': 'Sales_Administrative',
    'Other-service': 'Service_Care',
    'Machine-op-inspct': 'Skilled_Technical',
    'Missing': 'Unclassified Occupations',
    'Transport-moving': 'Skilled_Technical',
    'Handlers-cleaners': 'Service_Care',
    'Farming-fishing': 'Service_Care',
    'Tech-support': 'Skilled_Technical',
    'Protective-serv': 'Professional_Managerial',
    'Priv-house-serv': 'Service_Care',
    'Armed-Forces': 'Unclassified Occupations',
})

# Check value counts of new occupation column
df.drop(['occupation'], axis=1,inplace=True)

categorical_columns = ["work_class", "marital_status", "new_occupation", "relationship", "race", "sex"]

transformer = make_column_transformer(
    (OneHotEncoder(), categorical_columns),
    remainder='passthrough', verbose_feature_names_out=False)

transformed = transformer.fit_transform(df)
df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

X = df

# scale_01 = ["age", "final_weight", "education_num", "capitol_gain", "capitol_loss", "hours_per_week"]
scale_01 = ["age", "final_weight", "education_num", "capitol_loss", "hours_per_week"]
scaler = MinMaxScaler()
scaler.fit(X[scale_01])
X[scale_01] = scaler.transform(X[scale_01])

X['target'] = Y
X.to_csv('datasets/adult/adult-cleaned.csv', index=False)


