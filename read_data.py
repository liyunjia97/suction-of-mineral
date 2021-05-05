# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import preprocessing
def read_data():
    '''
    读取CFD处理之后的数据
    :return: 返回归一化后的DF数据，DF数据，纵向移动数据和横向移动数据
    '''
    dfL=pd.read_csv('D:\WORK_DATA_F\suction of mineral\L整体数据0414修改.csv',index_col=0)
    dfL.columns=['y/d','x/d','CFr','CFv']
    dfL=dfL[['x/d','y/d','CFr','CFv']]
    dfL['y/d']=dfL['y/d']+0.5
    dfH=pd.read_csv('D:\WORK_DATA_F\suction of mineral\H整体数据.csv',index_col=0)
    dfH['y/d'] = dfH['y/d'] + 0.5
    DF=pd.concat([dfH,dfL])
    DF['CFr']=-DF['CFr']
    scale=preprocessing.MinMaxScaler().fit(DF)
    predf=scale.transform(DF)
    df_minmax=pd.DataFrame(predf)
    df_minmax.columns=['x/d','y/d','CFr','CFv']
    return df_minmax,DF,dfL,dfH