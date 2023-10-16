import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("fraudTrain.csv")
y = data.iloc[:, -1].values


# Dropping irrelevant features
columns_to_drop = ["Unnamed: 0","cc_num","first","last","street","merchant","trans_num","unix_time","zip"]
data.drop(columns=columns_to_drop,inplace=True)



# dealing with date features
data['dob'] = pd.to_datetime(data['dob'])
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['trans_date_trans_time'] = data['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
# Get transaction month and year only
data["trans_month"] = data["trans_date_trans_time"].dt.month
data["trans_year"] = data["trans_date_trans_time"].dt.year

# turning dob feature into age
data["age"] = (data["trans_date_trans_time"]- data["dob"]).apply(lambda x: int(x.days / 365))

# getting distance bewteen seller and home
data['latitudinal_distance'] = abs(round(data['merch_lat'] - data['lat'],3))
data['longitudinal_distance'] = abs(round(data['merch_long'] - data['long'],3))
# dropping un-needed columns
drop_columns = ['trans_date_trans_time','city','lat','long','job','dob','merch_lat','merch_long','state','is_fraud']
data.drop(columns=drop_columns,inplace=True)

# categorical feature encoding
data = pd.get_dummies(data , columns=['category'] , prefix='category')
le = LabelEncoder()
data["gender"] = le.fit_transform(data["gender"])
# loading values into feature dataframe
x = data.iloc[:, :].values





