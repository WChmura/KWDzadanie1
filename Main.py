import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

boston_market_data = load_boston()
#print(boston_market_data['DESCR'])
#boston_market_data_p = pd.DataFrame(boston_market_data.data, columns=[boston_market_data.feature_names])
#boston_market_data_p.head()
#print(boston_market_data_p.describe())
scaler = StandardScaler()
boston_market_data['data'] = scaler.fit_transform(boston_market_data['data'])

