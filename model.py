import pandas as pd
import numpy as np

import sys
import os
home_directory_path = os.path.expanduser('~')
sys.path.append(home_directory_path +'/utils')

from prepare_utils import split_data

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.inspection import permutation_importance

from sklearn.preprocessing import MinMaxScaler


def split_mlb_data(tm_batting=pd.read_csv('data/team_batting.csv'), 
                   tm_pitching=pd.read_csv('data/team_pitching.csv')):
    
    # split batting data
    tm_batting_train,\
        tm_batting_validate,\
            tm_batting_test = split_data(tm_batting, validate_size=.15, test_size=.15, random_state=123)
    
    # split pitching data
    tm_pitching_train = tm_pitching.loc[tm_batting_train.index]
    tm_pitching_validate = tm_pitching.loc[tm_batting_validate.index]
    tm_pitching_test = tm_pitching.loc[tm_batting_test.index]
    
    # join pitching and batting stats
    train = pd.merge(left = tm_batting_train, right = tm_pitching_train, on=['year','Tm','W'], 
         suffixes= ("_bat", "_pit"))
    validate = pd.merge(left = tm_batting_validate, right = tm_pitching_validate, on=['year','Tm','W'], 
             suffixes= ("_bat", "_pit"))
    test = pd.merge(left = tm_batting_test, right = tm_pitching_test, on=['year','Tm','W'], 
             suffixes= ("_bat", "_pit"))
    
    return train, validate, test


def scale_mlb_data(train=split_mlb_data()[0],
                   validate=split_mlb_data()[1],
                   test=split_mlb_data()[2],
                   features=['OPS+', 'HR_bat', 'BatAge', 'OPS_pit', 'BB_pit', 'PAge'],
                   target=['W']):
    
    # separate features to scale
    X_train = train[features]
    X_validate = validate[features]
    X_test = test[features]

    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]
    
    # scale data
    scaler = MinMaxScaler()

    train_sc = pd.concat([pd.DataFrame(data=scaler.fit_transform(X_train),
                                       columns=X_train.columns), y_train],
                                       axis=1)
    validate_sc = pd.concat([pd.DataFrame(data=scaler.transform(X_validate),
                                          columns=X_validate.columns), y_validate],
                                      axis=1)
    test_sc = pd.concat([pd.DataFrame(data=scaler.transform(X_test), 
                                      columns=X_test.columns), y_train],
                                      axis=1)
    return train_sc, validate_sc, test_sc, scaler


def run_baseline_model(train, test, features, target):
    
    # split X and y
    X_train = train[features]
    X_test = test[features]

    y_train = train[target]
    y_test = test[target]
    
    # run model
    dummy = DummyRegressor().fit(X_train, y_train)    
    
    # RMSE
    train_rmse = mean_squared_error(y_train, dummy.predict(X_train), squared=False)
    test_rmse = mean_squared_error(y_test, dummy.predict(X_test), squared=False)
    # R2
    train_r2 = r2_score(y_train, dummy.predict(X_train))
    test_r2 = r2_score(y_test, dummy.predict(X_test))
    
    print(f'Train:\tRMSE = {round(train_rmse, 2)}\tR2 = {train_r2}')
    print(f'Test:\tRMSE = {round(test_rmse, 2)}\tR2 = {round(test_r2, 2)}')
    
    return train_rmse, train_r2, test_rmse, test_r2

def run_linear_model(train, test, features, target, scaler):
    
    # split X and y
    X_train = train[features]
    X_test = test[features]

    y_train = train[target]
    y_test = test[target]
    
    # run model
    lm = LinearRegression().fit(X_train, y_train)    
    
    # RMSE
    train_rmse = mean_squared_error(y_train, lm.predict(X_train), squared=False)
    test_rmse = mean_squared_error(y_test, lm.predict(X_test), squared=False)
    # R2
    train_r2 = r2_score(y_train, lm.predict(X_train))
    test_r2 = r2_score(y_test, lm.predict(X_test))
    
    print(f'Train:\tRMSE = {round(train_rmse, 2)}\tR2 = {round(train_r2, 2)}')
    print(f'Test:\tRMSE = {round(test_rmse, 2)}\tR2 = {round(test_r2, 2)}')
    
    df_scaled = pd.DataFrame(np.append(lm.coef_, lm.intercept_),
                             index=features + ['Intercept'],
                             columns=['Scaled Coeffs'])
    
    df_unscaled = pd.DataFrame(np.append(lm.coef_ * scaler.scale_, lm.intercept_),
                               index=features + ['Intercept'],
                               columns=['Unscaled Coeffs'])
    # Concatenate the two DataFrames horizontally
    result_df = pd.concat([df_scaled, df_unscaled], axis=1)

    # Display the combined DataFrame
    display(result_df)
    
    
    return train_rmse, train_r2, test_rmse, test_r2

def run_GBRegression_model(train, test, features, target):
    # split X and y
    X_train = train[features]
    X_test = test[features]

    y_train = train[target]
    y_test = test[target]
    
    # run model
#     gbr = GradientBoostingRegressor(random_state=123).fit(X_train, y_train)
#     gbr = GradientBoostingRegressor(learning_rate=0.03801806702724453, max_depth=6,
#                                       n_estimators=611, random_state=123,
#                                       subsample=0.12157377354410015).fit(X_train, y_train)
    gbr = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=500,
                                    random_state=123, subsample=0.5)\
                                    .fit(X_train, y_train)
    
    # RMSE
    train_rmse = mean_squared_error(y_train, gbr.predict(X_train), squared=False)
    test_rmse = mean_squared_error(y_test, gbr.predict(X_test), squared=False)
    # R2
    train_r2 = r2_score(y_train, gbr.predict(X_train))
    test_r2 = r2_score(y_test, gbr.predict(X_test))
    
    print(f'Train:\tRMSE = {round(train_rmse, 2)}\tR2 = {round(train_r2, 2)}')
    print(f'Test:\tRMSE = {round(test_rmse, 2)}\tR2 = {round(test_r2, 2)}')
     
#     # Plot feature importance
#     feature_importance = gbr.feature_importances_
#     sorted_idx = np.argsort(feature_importance)
#     pos = np.arange(sorted_idx.shape[0]) + 0.5
#     fig = plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.barh(pos, feature_importance[sorted_idx], align="center")
#     plt.yticks(pos, np.array(features)[sorted_idx])
#     plt.title("Feature Importance (MDI)")

#     result = permutation_importance(
#         gbr, X_test, y_test, n_repeats=10, random_state=123, n_jobs=2
#     )
#     sorted_idx = result.importances_mean.argsort()
#     plt.subplot(1, 2, 2)
#     plt.boxplot(
#         result.importances[sorted_idx].T,
#         vert=False,
#         labels=np.array(features)[sorted_idx],
#     )
#     plt.title("Permutation Importance (test set)")
#     fig.tight_layout()
#     plt.show()
    
    return train_rmse, train_r2, test_rmse, test_r2