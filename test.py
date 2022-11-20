import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from itertools import combinations
import matplotlib.pyplot as plt
import joblib

def max_min_diff(data):
    col_r = []
    ma_r = []
    for i in range(data.shape[0]):
        diff = max(data[i][7:46:2]) - min(data[i][7:46:2])
        ma_r.append(moving_a(data[i][7:46:2]))
        col_r.append(diff)
    col_r = np.array(col_r).reshape(len(col_r), -1)
    ma_r = np.array(ma_r)

    col_v = []
    ma_v = []
    for i in range(data.shape[0]):
        diff = max(data[i][8:47:2]) - min(data[i][8:47:2])
        col_v.append(diff)
        ma_v.append(moving_a(data[i][8:47:2]))
    col_v = np.array(col_v).reshape(len(col_v), -1)
    ma_v = np.array(col_v)
    return col_r, col_v, ma_r, ma_v

def moving_a(x, n=5, weighted = False):
    N = n
    if weighted:
        weights = np.exp(np.linspace(0,1,N))
        weights = weights/np.sum(weights)

    else:
        n = np.ones(N)
        weights = n/N
    x_new = np.convolve(weights,x)[N-1:-N+1]
    return x_new


def clean_data(X):
    i_v = []
    i_r = []
    for i in range(20):
        i_v.append(r'VOLUME_{}'.format(i+1))
        i_r.append(r'RET_{}'.format(i+1))

    x={}
    x[0]=X[i_v].T
    x[1]=X[i_r].T
    x[2]=X[['ID','DATE','STOCK','INDUSTRY','INDUSTRY_GROUP','SECTOR','SUB_INDUSTRY']].T

    for i in range(2):
        x[i]=(x[i].fillna(method="ffill")+x[i].fillna(method="bfill"))/2
        x[i].iloc[:,-1].fillna(method="ffill",inplace = True)
        x[i].iloc[:,0].fillna(method="bfill",inplace = True)
        x[i].fillna(0,inplace = True)
    x[2] = x[2].fillna(0)
    x_result = pd.concat([x[2],x[0],x[1]])
    return x_result.T

def weekly_means(data):
  meanx_train = data.copy()
  ret_cols = []
  for ret in meanx_train.columns:
    if "RET" in ret:
      ret_cols.append(ret)

  vol_cols = []
  for vol in meanx_train.columns:
    if "VOLUME" in vol:
      vol_cols.append(vol)

  list_1 = [0,5,10,15]
  list_2 =[5,10,15,20]

  for i in range(4):
    temp1 = ret_cols[list_1[i]:list_2[i]]
    temp2 = vol_cols[list_1[i]:list_2[i]]


    meanx_train['average R Week ' + str(i)] = meanx_train[temp1].mean(axis=1)
    meanx_train['average V Week ' + str(i)] = meanx_train[temp2].mean(axis=1)

  return pd.concat([meanx_train.iloc[:,:7],meanx_train.iloc[:,47:]],axis=1)

def data_permute(x_train):
    x_train = clean_data(x_train)
    col_r, col_v, ma_r, ma_v = max_min_diff(x_train.values)
    concat_diff = np.concatenate([col_r, col_v, ma_r, ma_v], axis=1)
    move_ave_name_r = [f'Move_ave_R_{i}' for i in range(1, ma_r.shape[1] + 1)]
    move_ave_name_v = [f'Move_ave_V_{i}' for i in range(1, ma_v.shape[1] + 1)]
    diff_df = pd.DataFrame(concat_diff, columns=['R_diff', 'V_diff'] + move_ave_name_r + move_ave_name_v)
    ema_ret = x_train.iloc[:, 7:46:2].ewm(com=0.4).mean()
    ema_ret.columns = [f'EMA_RET_{x}' for x in range(1, 21)]
    ema_vol = x_train.iloc[:, 8:47:2].ewm(com=0.4).mean()
    ema_vol.columns = [f'EMA_VOL_{x}' for x in range(1, 21)]
    x_train = pd.concat([x_train, diff_df], axis=1)
    x_train = pd.concat([x_train, ema_ret, ema_vol], axis=1)
    stats = ['mean', 'std']
    feature_to_gbs = ['INDUSTRY', 'SECTOR', 'STOCK', 'DATE', 'INDUSTRY_GROUP', 'SUB_INDUSTRY']
    feature_gbs = []
    for i in range(1, len(feature_to_gbs) + 1):
        feature_gbs += list(combinations(x_train[feature_to_gbs], i))
    meanx_train = weekly_means(x_train)
    target_feature = 'average R Week'

    for index in range(4):
        for i in range(len(feature_gbs)):
            feat = f'{target_feature} {index}'
            tmp_name = '_'.join(feature_gbs[i])
            for s in stats:
                meanx_train[f'{target_feature}_{index}_{tmp_name}_{s}'] = meanx_train.groupby(list(feature_gbs[i]))[
                feat].transform(s)
    meanx_train = meanx_train.dropna(axis=1)
    return meanx_train

# %%
x_train = pd.read_pickle("~/Downloads/x_train.pkl")
y_train = pd.read_pickle("~/Downloads/y_train.pkl")

x_train = data_permute(x_train)
features = x_train.columns.values
# %%
rf_params = {
    'n_estimators': 600,
    'max_depth': 2**3,
    'random_state': 0,
    'n_jobs': -1
}


model = RandomForestClassifier(**rf_params)
model.fit(x_train.drop(['ID', 'DATE'], axis=1), np.ravel(y_train.iloc[:, 1]))
# %%
importances = model.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# %%
new_features = []
for i in indices[::-1][:14]:
    if (features[i] != 'ID') and (features[i] != 'DATES'):
        new_features.append(features[i])
# %%
rf_params = {
    'n_estimators': 600,
    'max_depth': 2**3,
    'random_state': 0,
    'n_jobs': -1
}
new_model = RandomForestClassifier(**rf_params)
new_model.fit(x_train[new_features], np.ravel(y_train.iloc[:, 1]))

# %%
new_importances = new_model.feature_importances_
new_indices = np.argsort(new_importances)

plt.title('Feature Importances')
plt.barh(range(len(new_indices)), new_importances[new_indices], color='b', align='center')
plt.yticks(range(len(new_indices)), [new_features[i] for i in new_indices])
plt.xlabel('Relative Importance')
plt.show()

# %%
target = 'RET'

x_test = pd.read_pickle("~/Downloads/x_test.pkl")

X_test = data_permute(x_test)

y_pred = new_model.predict(X_test[new_features])


submission = pd.Series(y_pred)
submission.index = x_test['ID']
submission.name = target

submission.to_csv('~/Documents/Pycharm/y_pred_rforest(7).csv', index=True, header=True)

# %%
joblib.dump(model, "./random_forest(best).joblib")