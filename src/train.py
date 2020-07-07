import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

#read the data
X_full = pd.read_csv('input/train.csv', index_col='Id')
X_test_full = pd.read_csv('input/test.csv', index_col='Id')

#Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea','YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'OverallCond','PoolArea']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

#Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

#Define the model
model_1 = RandomForestRegressor(n_estimators=180, criterion='mae', random_state=0)

model_1.fit(X_train,y_train)
preds = model_1.predict(X_valid)

mae = mean_absolute_error(y_valid, preds)

print('MAE %d'%(mae))

#Fit model to the training data
model_1.fit(X,y)

#Generate test predictions 
preds_test = model_1.predict(X_test)

output = pd.DataFrame({'Id':X_test.index, 
        'SalePrice':preds_test})

output.to_csv('submission.csv', index=False)

