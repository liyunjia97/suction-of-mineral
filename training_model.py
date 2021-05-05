# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,max_error
from hyperopt import fmin, tpe, hp,Trials,partial
from keras import losses
from keras.layers.core import Dense
from keras.layers import  Input
from keras.models import Model
from keras.optimizers import Adam

def  Bayesian_optimize_poly(df_minmax,force):
    '''
    贝叶斯优化具体教程见链接：https://github.com/FontTian/hyperopt-doc-zh/wiki/FMin
    :param df_minmax: 归一化后的DF数据
    :param force: 径向力还是纵向力
    :return: 返回多项式回归超参数筛选后的最优超参数
    '''
    space = {"param_degree": hp.randint("param_degree", 5, 15)} #贝叶斯优化的搜索域
    ref_list = [pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[0].name,
                pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[1].name,
                pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[2].name,
                pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[3].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[0].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[1].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[2].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[3].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[4].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[5].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[6].name]

    def Polynomia_func(argsDict):
        model = Pipeline(
            [('poly', PolynomialFeatures(degree=argsDict["param_degree"])), ('linear', LinearRegression())])
        mse_list = []
        for i in range(len(ref_list)):
            if i < 4:
                df1 = df_minmax[df_minmax['x/d'] != ref_list[i]]
                df2 = df_minmax[df_minmax['x/d'] == ref_list[i]]
                X_data = np.array(df1[['x/d', 'y/d']])
                y_data = np.array(df1[[force]]).flatten()
                X_test = np.array(df2[['x/d', 'y/d']])
                y_test = np.array(df2[[force]]).flatten()
                X_data = X_data.astype(np.float32)
                y_data = y_data.astype(np.float32)
                X_test = X_test.astype(np.float32)
                y_test = y_test.astype(np.float32)
                model.fit(X_data, y_data)
                y_test_predict = model.predict(X_test)
                mse = mean_squared_error(y_test, y_test_predict)
                mse_list.append(mse)
            if i >= 4:
                df1 = df_minmax[df_minmax['y/d'] != ref_list[i]]
                df2 = df_minmax[df_minmax['y/d'] == ref_list[i]]
                X_data = np.array(df1[['x/d', 'y/d']])
                y_data = np.array(df1[[force]]).flatten()
                X_test = np.array(df2[['x/d', 'y/d']])
                y_test = np.array(df2[[force]]).flatten()
                X_data = X_data.astype(np.float32)
                y_data = y_data.astype(np.float32)
                X_test = X_test.astype(np.float32)
                y_test = y_test.astype(np.float32)
                model.fit(X_data, y_data)
                y_test_predict = model.predict(X_test)
                mse = mean_squared_error(y_test, y_test_predict)
                mse_list.append(mse)
        #     print(np.mean(mse_list),mse_list)
        return np.mean(mse_list)

    trials = Trials()
    algo = partial(tpe.suggest, n_startup_jobs=1)
    best = fmin(Polynomia_func, space, algo=algo, max_evals=100, trials=trials)
    return best
def Bayesian_optimize_nn(df_minmax,force):
    '''
    :param df_minmax: 归一化后的DF数据
    :param force: 径向力还是纵向力
    :return: 返回神经网络模型超参数筛选后的最优超参数
    '''
    space = {'units1': hp.choice('units1', [16, 64, 128, 320, 512]),
             'units2': hp.choice('units2', [16, 64, 128, 320, 512]),
             'units3': hp.choice('units3', [16, 64, 128, 320, 512]),
             'lr': hp.choice('lr', [0.01, 0.001, 0.0001]),
             'activation': hp.choice('activation', ['relu',
                                                    'sigmoid',
                                                    'tanh',
                                                    'linear']),
             'loss': hp.choice('loss', [losses.logcosh,
                                        losses.mse,
                                        losses.mae,
                                        losses.mape])}
    ref_list = [pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[0].name,
                pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[1].name,
                pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[2].name,
                pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[3].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[0].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[1].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[2].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[3].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[4].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[5].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[6].name]
    def experiment(params):
        main_input = Input(shape=(2,), name='main_input')
        x = Dense(params['units1'], activation=params['activation'])(main_input)
        x = Dense(params['units2'], activation=params['activation'])(x)
        x = Dense(params['units3'], activation=params['activation'])(x)
        output = Dense(1, activation="linear", name="out")(x)
        final_model = Model(inputs=[main_input], outputs=[output])
        opt = Adam(lr=params['lr'])
        final_model.compile(optimizer=opt, loss=params['loss'])

        mse_list = []
        for i in range(len(ref_list)):
            if i < 4:
                df1 = df_minmax[df_minmax['x/d'] != ref_list[i]]
                df2 = df_minmax[df_minmax['x/d'] == ref_list[i]]
                X_data = np.array(df1[['x/d', 'y/d']])
                y_data = np.array(df1[[force]]).flatten()
                X_test = np.array(df2[['x/d', 'y/d']])
                y_test = np.array(df2[[force]]).flatten()
                X_data = X_data.astype(np.float32)
                y_data = y_data.astype(np.float32)
                X_test = X_test.astype(np.float32)
                y_test = y_test.astype(np.float32)
                history = final_model.fit(X_data, y_data,
                                          epochs=30,
                                          batch_size=256,
                                          verbose=0,
                                          validation_data=(X_test, y_test),
                                          shuffle=True)
                y_test_predict = final_model.predict(X_test)
                mse = mean_squared_error(y_test, y_test_predict)
                mse_list.append(mse)
            if i >= 4:
                df1 = df_minmax[df_minmax['y/d'] != ref_list[i]]
                df2 = df_minmax[df_minmax['y/d'] == ref_list[i]]
                X_data = np.array(df1[['x/d', 'y/d']])
                y_data = np.array(df1[[force]]).flatten()
                X_test = np.array(df2[['x/d', 'y/d']])
                y_test = np.array(df2[[force]]).flatten()
                X_data = X_data.astype(np.float32)
                y_data = y_data.astype(np.float32)
                X_test = X_test.astype(np.float32)
                y_test = y_test.astype(np.float32)
                history = final_model.fit(X_data, y_data,
                                          epochs=30,
                                          batch_size=256,
                                          verbose=0,
                                          validation_data=(X_test, y_test),
                                          shuffle=True)
                y_test_predict = final_model.predict(X_test)
                mse = mean_squared_error(y_test, y_test_predict)
                mse_list.append(mse)

        mse = np.mean(mse_list)
        print('mse', mse)
        return mse
    algo = partial(tpe.suggest, n_startup_jobs=1)
    best = fmin(experiment, space, algo=algo, max_evals=200)
    return best
def polynomial_model_built():
    '''
    该函数是在贝叶斯优化得到最优超参数后，通过多项式回归建立模型
    :return: 返回建立的多项式回归模型
    '''
    model = Pipeline([('poly', PolynomialFeatures(degree=8)), ('linear', LinearRegression())])
    return model
def nn_model_built_Fr():
    '''
    该函数是在贝叶斯优化得到最优超参数后，通过人工神经网络建立模型
    :return: 返回建立的人工神经网络模型
    '''
    main_input = Input(shape=(2,), name='main_input')
    x = Dense(320, activation='tanh')(main_input)
    x = Dense(64, activation='tanh')(x)
    x = Dense(64, activation='tanh')(x)
    output = Dense(1, activation="linear", name="out")(x)
    final_model = Model(inputs=[main_input], outputs=[output])
    opt = Adam(lr=0.001)
    final_model.compile(optimizer=opt, loss=losses.mse)
    return final_model
def nn_model_built_Fv():
    '''
    该函数是在贝叶斯优化得到最优超参数后，通过人工神经网络建立模型
    :return: 返回建立的人工神经网络模型
    '''
    main_input = Input(shape=(2,), name='main_input')
    x = Dense(320, activation='relu')(main_input)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation="linear", name="out")(x)
    final_model = Model(inputs=[main_input], outputs=[output])
    opt = Adam(lr=0.0001)
    final_model.compile(optimizer=opt, loss=losses.mse)
    return final_model
def training_11_fold(DF,df_minmax,force,machine_model):
    '''
    该函数是对cfd处理后的数据进行十一折交叉验证并计算
    return:返回十一折交叉验证计算后的测试集的数据结果。以及各个评价指标
    '''
    DF_test = pd.DataFrame()
    CFr_max = DF[force].max()
    CFr_min = DF[force].min()
    ref_list = [pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[0].name,
                pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[1].name,
                pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[2].name,
                pd.DataFrame(df_minmax['x/d'].value_counts()).iloc[3].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[0].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[1].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[2].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[3].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[4].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[5].name,
                pd.DataFrame(df_minmax['y/d'].value_counts()).iloc[6].name]
    if machine_model=='polynomial':
        model=polynomial_model_built()
    elif machine_model=='nn':
        if force=='CFv':
            final_model=nn_model_built_Fv()
        elif force=='CFr':
            final_model=nn_model_built_Fr()
    else:
        return '模型输入错误'
    mse_data_list = []
    mae_data_list = []
    me_data_list = []
    mse_test_list = []
    mae_test_list = []
    me_test_list = []
    for i in range(len(ref_list)):
        if i < 4:
            df1 = df_minmax[df_minmax['x/d'] != ref_list[i]]
            df2 = df_minmax[df_minmax['x/d'] == ref_list[i]]
            X_data = np.array(df1[['x/d', 'y/d']])
            y_data = np.array(df1[[force]]).flatten()
            X_test = np.array(df2[['x/d', 'y/d']])
            y_test = np.array(df2[[force]]).flatten()
            X_data = X_data.astype(np.float32)
            y_data = y_data.astype(np.float32)
            X_test = X_test.astype(np.float32)
            y_test = y_test.astype(np.float32)
            if machine_model=='polynomial':
                model.fit(X_data, y_data)
                y_test_predict = model.predict(X_test)
                y_data_predict = model.predict(X_data)
            elif machine_model=='nn':
                history = final_model.fit(X_data, y_data,
                                          epochs=30,
                                          batch_size=256,
                                          verbose=0,
                                          validation_data=(X_test, y_test),
                                          shuffle=True)
                y_test_predict = final_model.predict(X_test)
                y_data_predict = final_model.predict(X_data)
            # print(pd.DataFrame(X_test))
            # print(pd.Series(y_test_predict.flatten()))
            df_test = pd.concat([pd.DataFrame(X_test), pd.Series(y_test_predict.flatten())], axis=1)
            DF_test = DF_test.append(df_test)

            y_test_predict = (y_test_predict - CFr_min) / (CFr_max - CFr_min)
            y_data_predict = (y_data_predict - CFr_min) / (CFr_max - CFr_min)
            y_data = (y_data - CFr_min) / (CFr_max - CFr_min)
            y_test = (y_test - CFr_min) / (CFr_max - CFr_min)

            mse_data = mean_squared_error(y_data, y_data_predict)
            mae_data = mean_absolute_error(y_data, y_data_predict)
            me_data = max_error(y_data, y_data_predict)
            mse_test = mean_squared_error(y_test, y_test_predict)
            mae_test = mean_absolute_error(y_test, y_test_predict)
            me_test = max_error(y_test, y_test_predict)
            mse_data_list.append(mse_data)
            mae_data_list.append(mae_data)
            me_data_list.append(me_data)
            mse_test_list.append(mse_test)
            mae_test_list.append(mae_test)
            me_test_list.append(me_test)
        if i >= 4:
            df1 = df_minmax[df_minmax['y/d'] != ref_list[i]]
            df2 = df_minmax[df_minmax['y/d'] == ref_list[i]]
            X_data = np.array(df1[['x/d', 'y/d']])
            y_data = np.array(df1[[force]]).flatten()
            X_test = np.array(df2[['x/d', 'y/d']])
            y_test = np.array(df2[[force]]).flatten()
            X_data = X_data.astype(np.float32)
            y_data = y_data.astype(np.float32)
            X_test = X_test.astype(np.float32)
            y_test = y_test.astype(np.float32)
            if machine_model=='polynomial':
                model.fit(X_data, y_data)
                y_test_predict = model.predict(X_test)
                y_data_predict = model.predict(X_data)
            elif machine_model=='nn':
                history = final_model.fit(X_data, y_data,
                                          epochs=30,
                                          batch_size=256,
                                          verbose=0,
                                          validation_data=(X_test, y_test),
                                          shuffle=True)
                y_test_predict = final_model.predict(X_test)
                y_data_predict = final_model.predict(X_data)
            df_test = pd.concat([pd.DataFrame(X_test), pd.Series(y_test_predict.flatten())], axis=1)
            DF_test = DF_test.append(df_test)
            y_test_predict = (y_test_predict - CFr_min) / (CFr_max - CFr_min)
            y_data_predict = (y_data_predict - CFr_min) / (CFr_max - CFr_min)
            y_data = (y_data - CFr_min) / (CFr_max - CFr_min)
            y_test = (y_test - CFr_min) / (CFr_max - CFr_min)
            mse_data = mean_squared_error(y_data, y_data_predict)
            mae_data = mean_absolute_error(y_data, y_data_predict)
            me_data = max_error(y_data, y_data_predict)
            mse_test = mean_squared_error(y_test, y_test_predict)
            mae_test = mean_absolute_error(y_test, y_test_predict)
            me_test = max_error(y_test, y_test_predict)
            mse_data_list.append(mse_data)
            mae_data_list.append(mae_data)
            me_data_list.append(me_data)
            mse_test_list.append(mse_test)
            mae_test_list.append(mae_test)
            me_test_list.append(me_test)
    DF_test.columns = ['x/d', 'y/d', force]
    mse_data_mean=np.mean(mse_data_list)
    mae_data_mean=np.mean(mae_data_list)
    me_data_mean=np.mean(me_data_list)
    mse_test_mean=np.mean(mse_test_list)
    mae_test_mean=np.mean(mae_test_list)
    me_test_mean=np.mean(me_test_list)
    return DF_test,mse_data_mean,mae_data_mean,me_data_mean,mse_test_mean,mae_test_mean,me_test_mean