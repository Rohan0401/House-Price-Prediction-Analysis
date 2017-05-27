#trial
import pandas as pd
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition'])) 
import numpy as np
from scipy.stats import skew
import matplotlib
matplotlib.use('Agg')#为什么有这一步？
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index  
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) 
skewed_feats = skewed_feats[skewed_feats > 0.75]  
skewed_feats = skewed_feats.index 
all_data[skewed_feats] = np.log1p(all_data[skewed_feats]) 
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

import xgboost as xgb
#如何得到下面XGBRegressor里面的参数？
regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
label_df = pd.DataFrame(index = train.index, columns=["SalePrice"])
#label_df是经过log化的train set中的SalePrice的数据
label_df["SalePrice"] = np.log(train["SalePrice"])
regr.fit(X_train, label_df)
y_pred = regr.predict(X_train)
y_test = label_df

from sklearn.metrics import mean_squared_error
#此处定义的这个求误差方程rmse哪里用到了？
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
y_pred_xgb = regr.predict(X_test)

from sklearn.linear_model import Lasso
#此处需要用cross-validation来找到这个best_alpha
best_alpha = 0.00099
regr = Lasso(alpha=best_alpha, max_iter=50000)
regr.fit(X_train, label_df)
y_pred = regr.predict(X_train)
y_test = label_df
y_pred_lasso = regr.predict(X_test)

#取XGBoost和Lasso两个模型的平均值为最终的预测结果，前面对于y使用了log化，此处把y转换回去
y_pred = (y_pred_xgb + y_pred_lasso) / 2
y_pred = np.exp(y_pred)

pred_df = pd.DataFrame(y_pred, index=test["Id"], columns=["SalePrice"])
pred_df.to_csv('output.csv', header=True, index_label='Id')