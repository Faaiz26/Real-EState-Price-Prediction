##################day 1 ###################################
import pandas as pd
df1 = pd.read_csv('bengaluru_house_prices.csv')
print(df1)
df1.shape
df1.groupby('area_type')['area_type'].agg('count')
# ndropping not needed value
df2 = df1.drop(['area_type','society','balcony','availability'],axis = 'columns')
df2
# data cleaning process
df2.isnull().sum()
df3 = df2.dropna()
df3.isnull().sum()
df3['size'].unique()

df3['bhk'] = df3['size'].apply(lambda x : int(x.split(' ' )[0]))
df3.bhk
df3['bhk'].unique()
df3[df3['bhk']>20]
df3
# now checking  total_qft value
df3.total_sqft.unique()

def is_float(x):
    try:
        float(x)
    except:
        return(False)
    return(True)
df3[~df3['total_sqft'].apply(is_float)]


def convert_sqft_to_num (x):
    tokens = x.split('-')
    if len(tokens) == 2 :
        return(float(tokens[0]) + float(tokens[1]) )/2
    try: 
        return(float(x))
    except:
        return(None)
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.loc[30]
###############################################day1 end #########################################
###day 2

df5 = df4.copy()
df5.head()
##feature engineering
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft'] 
df5['price_per_sqft']
df5.location.unique()
len(df5.location.unique())
df5.location = df5.location.apply(lambda x :x.strip())

location_stats = df5.groupby('location')['location'].agg('count')
location_stats.sort_values(ascending = False)
len(location_stats[location_stats <= 10])
location_stats_less_than_10 = location_stats[location_stats <= 10]
location_stats_less_than_10.sort_values(ascending = False)
len(df5.location.unique())
df5.location = df5.location.apply(lambda x : 'other' if x  in location_stats_less_than_10 else x )
len(df5.location.unique())
#####################################################################################################
df5.head(10)
df5.shape
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6['price_per_sqft'].describe()

import numpy as np
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index = True)
    return(df_out)
df7 = remove_pps_outliers(df6)
df7.shape

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                    "mean" : np.mean(bhk_df.price_per_sqft),
                    "std": np.std(bhk_df.price_per_sqft),
                    "count" : bhk_df.shape[0]}
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats ['count'] > 5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices,axis = "index")
df8 = remove_bhk_outliers(df7)
df8.shape
df8.bath.unique()
df8[df8.bath > 10]
df8[df8.bath >df8.bhk+2]  # outlier

df9 = df8[df8.bath < df8.bhk+2]
df9.shape
#################################################################################################################################################
#model building
df10 = df9.drop(['size','price_per_sqft'], axis= 'columns')
df10.head()
# one hot encoding
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10,dummies.drop('other',axis = 'columns')],axis ='columns')
df11.head()
df12 = df11.drop('location',axis = 'columns')
df12.head()
# creating training variable
X = df12.drop('price',axis = 'columns')
X.head()
Y = df12.price
Y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,Y_train)
lr_clf.score(X_test,Y_test)

#validating with k fold cross validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits= 5 ,test_size = 0.2 ,random_state = 0)
cross_val_score(LinearRegression(),X,Y,cv=cv)

#grid search cv
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor




def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,Y)


def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]
predict_price('1st Phase JP Nagar',1000,2,2)
predict_price('1st Phase JP Nagar',1000,2,4)  
predict_price('Indira Nagar',1000,2,2)


# exporting model in flask file
import pickle
with open ('banglore_home_price_model.pickle','wb')as f:
    pickle.dump(lr_clf,f)

import json
columns = {
        'data columns' : [col.lower() for col in X.columns]}
with open ('columns.json','w') as f:
    f.write(json.dumps(columns))







