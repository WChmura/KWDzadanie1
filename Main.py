import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

boston_market_data = load_boston()
#print(boston_market_data['DESCR'])
#boston_market_data_p = pd.DataFrame(boston_market_data.data, columns=[boston_market_data.feature_names])
#boston_market_data_p.head()
#print(boston_market_data_p.describe())
scaler = StandardScaler()
boston_market_data['data'] = scaler.fit_transform(boston_market_data['data'])
boston_train_data, boston_test_data, boston_train_target, boston_test_target = train_test_split(boston_market_data['data'], boston_market_data['target'], test_size=0.05)
print("boston_train_data:", boston_train_data.shape)
print("boston_test_data:", boston_test_data.shape)
linear_regression = LinearRegression()
linear_regression.fit(boston_train_data, boston_train_target)
id=5
linear_regression_prediction = linear_regression.predict(boston_test_data[id,:].reshape(1,-1))
print("Model predicted for house number {0} value {1}".format(id, linear_regression_prediction))
print("Real value for house number \"{0}\" is {1}".format(id, boston_test_target[id]))
print('Coefficients of a learned model: \n', linear_regression.coef_)

linear_regression_predictions = linear_regression.predict(boston_test_data)
print("Linear regression predictions=", end="")
print(linear_regression_predictions)
print("Mean squared error of a learned model: %.2f" % mean_squared_error(boston_test_target, linear_regression.predict(boston_test_data)))
print('Variance score: %.2f' % r2_score(boston_test_target, linear_regression.predict(boston_test_data)))
scores = cross_val_score(LinearRegression(), boston_market_data['data'], boston_market_data['target'], cv=4)
print("cross validation =", end="")
print(scores)