import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
# read the data
housing = pd.read_csv('housing.csv')
#creating a stratified dataset
housing['income_cat']=pd.cut(housing['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
labels=[1,2,3,4,5])
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['income_cat']):
   strat_train_set=housing.loc[train_index].drop('income_cat',axis=1)
   strat_test_set=housing.loc[test_index].drop('income_cat',axis=1)
   #working on thoe copy of data
   housing=strat_train_set.copy()
   #separate features and labels
   housing_labels=housing['median_house_value'].copy()
   housing=housing.drop('median_house_value',axis=1)
   print(housing,housing_labels)
   #separate numerical and categorial columns
  # Define numerical and categorical attributes
num_attribs = housing.drop('ocean_proximity', axis=1).columns.tolist()
cat_attribs = ['ocean_proximity']

# Pipeline for numerical columns
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical columns
cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Constructing the full pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])

# Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
#train the model
#linear regression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_preds=lin_reg.predict(housing_prepared)
lin_rmse=-cross_val_score(lin_reg,housing_prepared,housing_labels,scoring='neg_root_mean_squared_error',cv=10)
print('the root_mean_sqaured linear regression is')
print(pd.Series(lin_rmse).describe())
#random forest model
random_forest_reg=RandomForestRegressor()
random_forest_reg.fit(housing_prepared,housing_labels)
random_forest_preds=random_forest_reg.predict(housing_prepared)
random_forest_rmse=-cross_val_score(random_forest_reg,housing_prepared,housing_labels,scoring='neg_root_mean_squared_error',cv=10)
print('the root_mean_sqaured random_forest is')
print(pd.Series(random_forest_rmse).describe())
#Decision tree model using cross validation
Desc_reg=DecisionTreeRegressor()
Desc_reg.fit(housing_prepared,housing_labels)
Desc_preds=Desc_reg.predict(housing_prepared)
Desc_rmses=-cross_val_score(Desc_reg,housing_prepared,housing_labels,scoring='neg_root_mean_squared_error',cv=10)
print('the root_mean_sqaured decision_tree is')
print(pd.Series(Desc_rmses).describe())