# -*- coding: utf-8 -*-
'''
将cfd数据进行预处理，以符合机器学习训练要求
在实际处理过程中应该分别对垂向移动和横向移动数据进行相应的处理。
在运行这个代码的时候请每次只运行 #%% 符号里面的数据
'''
import pandas as pd
import numpy as np
import pywt
#%%
# 纵向力的处理
#该代码是以L80为例
vertical_df1=pd.read_csv(r'原始数据\垂向移动\L80d40V1v2_1.5_31.5.csv')
vertical_df1.columns=['time','Fx','Fz','Fy']
vertical_df1.drop('Fz',axis=1,inplace=True)
vertical_df1=vertical_df1[vertical_df1['time']>list(vertical_df1['time'])[-1]-30].copy()
vertical_df1['h'] = vertical_df1['time'] + 50 - 1.5
vertical_df1['x/d'] = 2
vertical_df1['y/d'] =( vertical_df1['h']-20)/40
#CFr是横向力，Cfv是垂向力
vertical_df1['CFr']=vertical_df1['Fx']/(1/2 *1000*2**2 * np.pi * 0.02**2)
vertical_df1['CFv']=(vertical_df1['Fy']-0.328736255)/(1/2 *1000*2**2 * np.pi * 0.02**2)
vertical_df1.drop(['time','Fx','Fy','h'],axis=1,inplace=True)
vertical_df1.to_csv(r'原始数据\垂向移动\L80d40V1v2_1.5_31.5_change.csv')

#%%
#径向力的处理
#该代码是以H80为例
broad_wise_df1=pd.read_csv(r'原始数据\横向移动\H80d40V2.5v2.csv')
broad_wise_df1.columns=['time','Fx','Fz','Fy']
broad_wise_df1.drop('Fz',axis=1,inplace=True)
broad_wise_df1['distance']=150-broad_wise_df1['time']*2.5
broad_wise_df1['x/d'] = broad_wise_df1['distance']/40
broad_wise_df1['y/d'] =(80-20)/40#对于不同的高度要更改这个值
broad_wise_df1['Fy-buoyancy']=broad_wise_df1['Fy']-0.328736255

broad_wise_df1['CFr']=broad_wise_df1['Fx']/(1/2 *1000*2**2 * np.pi * 0.02**2)
broad_wise_df1['CFv']=(broad_wise_df1['Fy']-0.328736255)/(1/2 *1000*2**2 * np.pi * 0.02**2)
broad_wise_df1=broad_wise_df1[broad_wise_df1['time']<=60]
broad_wise_df1.sort_values(by='distance',inplace=True)
broad_wise_df1.to_csv(r'原始数据\横向移动\H80d40V2.5v2_change.csv')
#当经过以上代码将垂向数据和横向数据处理后，应该将数据进行对应位置的异常点删除，然后通过小波滤波进行重新过滤。
# 最后将所有的处理数据存放到 原始数据基本处理 文件夹内
#%%
# 读取所有的之前处理后的数据，然后进行滤波，并将数据保存到 L整体数据0414修改.csv 和 H整体数据.csv 里面。
H50=pd.read_csv(r'原始数据基本处理\H50d40V2.5v2_change1.csv',index_col=0)
H55=pd.read_csv(r'原始数据基本处理\H55d40V2.5v2_change1.csv',index_col=0)
H60=pd.read_csv(r'原始数据基本处理\H60d40V2.5v2_change1.csv',index_col=0)
H65=pd.read_csv(r'原始数据基本处理\H65d40V2.5v2_change1.csv',index_col=0)
H70=pd.read_csv(r'原始数据基本处理\H70d40V2.5v2_change1.csv',index_col=0)
H75=pd.read_csv(r'原始数据基本处理\H75d40V2.5v2_change1.csv',index_col=0)
H80=pd.read_csv(r'原始数据基本处理\H80d40V2.5v2_change1.csv',index_col=0)
L0=pd.read_csv(r'原始数据基本处理\L0d40V1v217.3_47.3_change.csv')
L20=pd.read_csv(r'原始数据基本处理\L20d40V1v2_1.5_31.5._change.csv')
L40=pd.read_csv(r'原始数据基本处理\L40d40V1v2_1.5_31.5_change.csv')
L60=pd.read_csv(r'原始数据基本处理\L60d40V1v2_1.5_31.5_change.csv')
L80=pd.read_csv(r'原始数据基本处理\L80d40V1v2_1.5_31.5_change.csv')
#小波滤波函数
def pywt_calculte(data1, wave, level):
    '''
    该函数是对cfd处理之后数据进行滤波得到的结果
    :param data1: 需要进行滤波的数据
    :param wave: 小波类型
    :param level: 小波级数
    :return: 返回值为滤波后的小波
    '''
    index = []
    data = []
    for i in range(len(data1)):
        X = float(i)
        Y = float(data1[i])
        index.append(X)
        data.append(Y)
    w = pywt.Wavelet(wave)  # 选用Daubechies8小波
    threshold = np.sqrt(2 *np.log10(len(data)))  #  定义阈值。np.sqrt(2*np.log10(len(data)))
    coeffs = pywt.wavedec(data, wave, level=level)  # 将信号进行小波分解
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波,默认是软阈值，https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
    datarec = pywt.waverec(coeffs, wave)  # 将信号进行小波重构
    #在选用小波滤波时，主要有三个参数需要调节或者选择，分别是小波类型，小波级数和小波阈值
    return datarec
#%%
# L0进行处理
data=pywt_calculte(np.array(L0['Cfv'][:1110]),'db8',6)
data1=pywt_calculte(np.array(L0['Cfv'][1180:]),'db8',6)
L0_finall=pd.DataFrame()
L0_finall['y/d']=list(L0['y/d'][:1110])+list(L0['y/d'][1180:])
L0_finall['x/d']=0
L0_finall['Cfr']=0
L0_finall['Cfv']=list(data)+list(data1)
#%%
# L20进行处理
data_cfr1=pywt_calculte(np.array(L20['Cfr'][:3205]),'db8',9)[:-1]
data_cfr2=pywt_calculte(np.array(L20['Cfr'][3550:]),'db8',9)[:-1]
data_cfv1=pywt_calculte(np.array(L20['Cfv'][:3205]),'db8',8)[:-1]
data_cfv2=pywt_calculte(np.array(L20['Cfv'][3550:]),'db8',8)[:-1]
L20_finall=pd.DataFrame()
L20_finall['y/d']=list(L20['y/d'][:3205])+list(L20['y/d'][3550:])
L20_finall['x/d']=list(L20['x/d'][:3205])+list(L20['x/d'][3550:])
L20_finall['Cfr']=list(data_cfr1)+list(data_cfr2)
L20_finall['Cfv']=list(data_cfv1)+list(data_cfv2)
#%%
# L40进行处理
list1=[i for i in range(933,1048)]
list2=[i for i in range(1725,1827)]
list3=[i for i in range(2514,2563)]
list_=[]
list_.extend(list1)
list_.extend(list2)
list_.extend(list3)
data_cfr1=pywt_calculte(np.array(L40['CFr'][:933]),'db8',8)[:-1]
data_cfr2=pywt_calculte(np.array(L40['CFr'][1048:1725]),'db8',8)[:-1]
data_cfr3=pywt_calculte(np.array(L40['CFr'][1827:2514]),'db8',8)[:-1]
data_cfr4=pywt_calculte(np.array(L40['CFr'][2563:]),'db8',8)
data_cfv1=pywt_calculte(np.array(L40['CFv'][:933]),'db8',8)[:-1]
data_cfv2=pywt_calculte(np.array(L40['CFv'][1048:1725]),'db8',8)[:-1]
data_cfv3=pywt_calculte(np.array(L40['CFv'][1827:2514]),'db8',8)[:-1]
data_cfv4=pywt_calculte(np.array(L40['CFv'][2563:]),'db8',8)
L40_finall=pd.DataFrame()
L40_finall['y/d']=L40['y/d'].drop(list_)
L40_finall['x/d']=L40['x/d'].drop(list_)
L40_finall['Cfr']=list(data_cfr1)+list(data_cfr2)+list(data_cfr3)+list(data_cfr4)
L40_finall['Cfv']=list(data_cfv1)+list(data_cfv2)+list(data_cfv3)+list(data_cfv4)
#%%
#L60进行处理
lista=[i for i in range(1692,1828)]
listb=[i for i in range(3143,3347)]
list__=[]
list__.extend(lista)
list__.extend(listb)
data_cfr1=pywt_calculte(np.array(L60['CFr'][:1692]),'db8',9)
data_cfr2=pywt_calculte(np.array(L60['CFr'][1828:3143]),'db8',9)[:-1]
data_cfr3=pywt_calculte(np.array(L60['CFr'][3347:]),'db8',9)
data_cfv1=pywt_calculte(np.array(L60['CFr'][:1692]),'db8',9)
data_cfv2=pywt_calculte(np.array(L60['CFr'][1828:3143]),'db8',9)[:-1]
data_cfv3=pywt_calculte(np.array(L60['CFr'][3347:]),'db8',9)
L60_finall=pd.DataFrame()
L60_finall['y/d']=L60['y/d'].drop(list__)
L60_finall['x/d']=L60['x/d'].drop(list__)
L60_finall['Cfr']=list(data_cfr1)+list(data_cfr2)+list(data_cfr3)
L60_finall['Cfv']=list(data_cfv1)+list(data_cfv2)+list(data_cfv3)
#%%
#L80进行处理
data_cfr1=pywt_calculte(np.array(L80['CFr'][:3069]),'db8',9)[:-1]
data_cfr2=pywt_calculte(np.array(L80['CFr'][3408:]),'db8',9)[:-1]
L80_finall=pd.DataFrame()
L80_finall['x/d']=L80['x/d'].drop([i for i in range(3069,3408)])
L80_finall['y/d']=L80['y/d'].drop([i for i in range(3069,3408)])
L80_finall['Cfr']=list(data_cfr1)+list(data_cfr2)
L80_finall['Cfv']=0
#%%
# L数据的保存
# pd.concat([L0_finall,L20_finall,L60_finall,L80_finall]).to_csv('L整体数据0414修改.csv')
#%%
def wave_data(data1, wave, threshold, level):
    global data,datarec
    ecg = data1
    index = []
    data = []
    for i in range(len(ecg)):
        X = float(i)
        Y = float(ecg[i])
        index.append(X)
        data.append(Y)
    w = pywt.Wavelet(wave)  # 选用Daubechies8小波
    # maxlev = pywt.dwt_max_level(len(data), w.dec_len)
#     print("maximum level is " + str(maxlev))
    threshold = np.sqrt(2 *
                        np.log10(len(data)))  #  np.sqrt(2*np.log10(len(data)))
    coeffs = pywt.wavedec(data, wave, level=level)  # 将信号进行小波分解
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(
            coeffs[i], threshold * max(coeffs[i])
        )  # 将噪声滤波,默认是软阈值，https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html

    datarec = pywt.waverec(coeffs, wave)  # 将信号进行小波重构
    mintime = 0
    maxtime = mintime + len(data) + 1
    return datarec[mintime:maxtime - 1]
#H的整体处理
H50_finall=pd.DataFrame()
H50_finall['x/d']=H50['x/d']
H50_finall['y/d']=H50['y/d']
H50_finall['CFr']=wave_data(np.array(H50['CFr']),'db8',np.sqrt(2 * np.log10(len(np.array(H50['CFr'])))),8)
H50_finall['CFv']=wave_data(np.array(H50['CFv']),'db8',np.sqrt(2 * np.log10(len(np.array(H50['CFv'])))),8)
H55_finall=pd.DataFrame()
H55_finall['x/d']=H55['x/d']
H55_finall['y/d']=H55['y/d']
H55_finall['CFr']=wave_data(np.array(H55['CFr']),'db8',np.sqrt(2 * np.log10(len(np.array(H55['CFr'])))),8)
H55_finall['CFv']=wave_data(np.array(H55['CFv']),'db8',np.sqrt(2 * np.log10(len(np.array(H55['CFv'])))),8)
H60_finall=pd.DataFrame()
H60_finall['x/d']=H60['x/d']
H60_finall['y/d']=H60['y/d']
H60_finall['CFr']=wave_data(np.array(H60['CFr']),'db8',np.sqrt(2 * np.log10(len(np.array(H60['CFr'])))),8)
H60_finall['CFv']=wave_data(np.array(H60['CFv']),'db8',np.sqrt(2 * np.log10(len(np.array(H60['CFv'])))),8)
H65_finall=pd.DataFrame()
H65_finall['x/d']=H65['x/d']
H65_finall['y/d']=H65['y/d']
H65_finall['CFr']=wave_data(np.array(H65['CFr']),'db8',np.sqrt(2 * np.log10(len(np.array(H65['CFr'])))),8)
H65_finall['CFv']=wave_data(np.array(H65['CFv']),'db8',np.sqrt(2 * np.log10(len(np.array(H65['CFv'])))),8)
H70_finall=pd.DataFrame()
H70_finall['x/d']=H70['x/d']
H70_finall['y/d']=H70['y/d']
H70_finall['CFr']=wave_data(np.array(H70['CFr']),'db8',np.sqrt(2 * np.log10(len(np.array(H70['CFr'])))),8)
H70_finall['CFv']=wave_data(np.array(H70['CFv']),'db8',np.sqrt(2 * np.log10(len(np.array(H70['CFv'])))),8)
H75_finall=pd.DataFrame()
H75_finall['x/d']=H75['x/d']
H75_finall['y/d']=H75['y/d']
H75_finall['CFr']=wave_data(np.array(H75['CFr']),'db8',np.sqrt(2 * np.log10(len(np.array(H75['CFr'])))),8)
H75_finall['CFv']=wave_data(np.array(H75['CFv']),'db8',np.sqrt(2 * np.log10(len(np.array(H75['CFv'])))),8)
H80_finall=pd.DataFrame()
H80_finall['x/d']=H80['x/d']
H80_finall['y/d']=H80['y/d']
H80_finall['CFr']=wave_data(np.array(H80['CFr']),'db8',np.sqrt(2 * np.log10(len(np.array(H80['CFr'])))),8)
H80_finall['CFv']=wave_data(np.array(H80['CFv']),'db8',np.sqrt(2 * np.log10(len(np.array(H80['CFv'])))),8)
# H数据的保存
# pd.concat([H50_finall,H55_finall,H60_finall,H65_finall,H70_finall,H75_finall,H80_finall]).to_csv('H整体数据.csv')