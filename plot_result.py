# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import pywt
from training_model import nn_model_built_Fv,nn_model_built_Fr,polynomial_model_built
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
def pltxd_test_result_contrast(DF,df_minmax,DF_test_polynomial,DF_test_nn,force):
    '''
    该函数是对径向移动的测试集数据进行预测值和真实值的对比。函数中的pltxd指的是横向移动的数据
    :param DF: 指的是将H整体数据.csv(是对数据进行滤波)和L整体数据0414修改.csv(是对数据进行滤波)进行合并之后的数据
    :param df_minmax:指的是将DF进行归一化之后的数据
    :param DF_test_polynomial: 指的是通过多项式回归十一折交叉验证所得到的测试集的数据
    :param DF_test_nn: 指的是人工神经网络十一折交叉验证所得到的测试集的数据
    :param force: 指的是径向力还是纵向力
    :return: 该函数无返回值，直接对交叉验证的预测值和原有值进行预测之后绘制的图形
    '''
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
    ref_list1 = [pd.DataFrame(DF_test_polynomial['x/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test_polynomial['x/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test_polynomial['x/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test_polynomial['x/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[4].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[5].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[6].name]
    ref_list2 = [pd.DataFrame(DF_test_nn['x/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test_nn['x/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test_nn['x/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test_nn['x/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[4].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[5].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[6].name]
    def inverse_transZ(x):
        max_ = DF[force].max()
        min_ = DF[force].min()
        return x * (max_ - min_) + min_
    def inverse_transX(x):
        max_ = DF['x/d'].max()
        min_ = DF['x/d'].min()
        return x * (max_ - min_) + min_
    def inverse_transY(x):
        max_ = DF['y/d'].max()
        min_ = DF['y/d'].min()
        return x * (max_ - min_) + min_
    plt.subplots(figsize=(10, 7))

    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[4]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[4]][[force]]))))), s=0.05, marker='D', color='darkorange',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[4]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[4]][[force]]))))), s=0.05, marker='D', color='cornflowerblue',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[4]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[4]][[force]]))))), s=0.05,marker='D', color='lawngreen',alpha=0.5)

    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[5]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[5]][[force]]))))), s=0.05, marker='D', color='darkorange',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[5]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[5]][[force]]))))), s=0.05, marker='D', color='cornflowerblue',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[5]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[5]][[force]]))))), s=0.05,marker='D', color='lawngreen',alpha=0.5)

    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[6]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[6]][[force]]))))), s=0.05, marker='D', color='darkorange',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[6]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[6]][[force]]))))), s=0.05, marker='D', color='cornflowerblue',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[6]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[6]][[force]]))))), s=0.05, marker='D', color='lawngreen',alpha=0.5)

    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[7]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[7]][[force]]))))), s=0.05, marker='D', color='darkorange',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[7]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[7]][[force]]))))), s=0.05, marker='D', color='cornflowerblue',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[7]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[7]][[force]]))))), s=0.05, marker='D', color='lawngreen',alpha=0.5)


    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[8]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[8]][[force]]))))), s=0.05, marker='D', color='darkorange',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[8]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[8]][[force]]))))), s=0.05, marker='D', color='cornflowerblue',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[8]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[8]][[force]]))))), s=0.05, marker='D', color='lawngreen',alpha=0.5)


    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[9]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[9]][[force]]))))), s=0.05, marker='D', color='darkorange',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[9]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[9]][[force]]))))), s=0.05, marker='D', color='cornflowerblue',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[9]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[9]][[force]]))))), s=0.05, marker='D', color='lawngreen',alpha=0.5)


    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[10]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[10]][[force]]))))), s=0.05, marker='D', color='darkorange',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[10]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_polynomial[DF_test_polynomial['y/d'] == ref_list1[10]][[force]]))))), s=0.05, marker='D', color='cornflowerblue',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[10]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_nn[DF_test_nn['y/d'] == ref_list2[10]][[force]]))))), s=0.05, marker='D', color='lawngreen',alpha=0.5)


    color = ['darkorange', 'cornflowerblue','lawngreen']
    labels = ['真实值', '多项式回归预测值','神经网络预测值']
    patches = [mpatches.Patch(facecolor=color[i], linewidth=0.02, label="{:s}".format(labels[i])) for i in
               range(len(color))]
    plt.legend(handles=patches, ncol=1,fontsize='x-large')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('x/d',fontsize=20)
    plt.ylabel(force,fontsize=20)
    plt.show()
def pltyd_test_result_contrast(DF,df_minmax,DF_test_polynomial,DF_test_nn,force):
    '''
    该函数是对纵向移动的测试集数据进行预测值和真实值的对比。其中函数中的pltyd指的是纵向移动的数据.
    :param DF: 指的是将H整体数据.csv(是对数据进行滤波)和L整体数据0414修改.csv(是对数据进行滤波)进行合并之后的数据
    :param df_minmax:指的是将DF进行归一化之后的数据
    :param DF_test_polynomial: 指的是通过多项式回归十一折交叉验证所得到的测试集的数据
    :param DF_test_nn: 指的是人工神经网络十一折交叉验证所得到的测试集的数据
    :param force: 指的是径向力还是纵向力
    :return: 该函数无返回值，直接对交叉验证的预测值和原有值进行预测之后绘制的图形
    '''
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
    ref_list1 = [pd.DataFrame(DF_test_polynomial['x/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test_polynomial['x/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test_polynomial['x/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test_polynomial['x/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[4].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[5].name,
                 pd.DataFrame(DF_test_polynomial['y/d'].value_counts()).iloc[6].name]
    ref_list2 = [pd.DataFrame(DF_test_nn['x/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test_nn['x/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test_nn['x/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test_nn['x/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[4].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[5].name,
                 pd.DataFrame(DF_test_nn['y/d'].value_counts()).iloc[6].name]
    def inverse_transZ(x):
        max_ = DF[force].max()
        min_ = DF[force].min()
        return x * (max_ - min_) + min_
    def inverse_transX(x):
        max_ = DF['x/d'].max()
        min_ = DF['x/d'].min()
        return x * (max_ - min_) + min_
    def inverse_transY(x):
        max_ = DF['y/d'].max()
        min_ = DF['y/d'].min()
        return x * (max_ - min_) + min_
    plt.subplots(figsize=(10, 7))
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[0]][['y/d']]))))),
        np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[0]][[force]]))))),s=0.05, marker='D', color='darkorange',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test_polynomial[DF_test_polynomial['x/d'] == ref_list1[0]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_polynomial[DF_test_polynomial['x/d'] == ref_list1[0]][[force]]))))),s=0.05, marker='D', color='cornflowerblue',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test_nn[DF_test_nn['x/d'] == ref_list2[0]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_nn[DF_test_nn['x/d'] == ref_list2[0]][[force]]))))),s=0.05, marker='D', color='lawngreen',alpha=0.5)

    plt.scatter(np.array(list(map(inverse_transY, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[1]][['y/d']]))))),
        np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[1]][[force]]))))),s=0.05, marker='D', color='darkorange',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test_polynomial[DF_test_polynomial['x/d'] == ref_list1[1]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_polynomial[DF_test_polynomial['x/d'] == ref_list1[1]][[force]]))))),s=0.05, marker='D', color='cornflowerblue',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test_nn[DF_test_nn['x/d'] == ref_list2[1]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_nn[DF_test_nn['x/d'] == ref_list2[1]][[force]]))))),s=0.05, marker='D', color='lawngreen',alpha=0.5)


    plt.scatter(np.array(list(map(inverse_transY, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[2]][['y/d']]))))),
        np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[2]][[force]]))))),s=0.05, marker='D', color='darkorange',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test_polynomial[DF_test_polynomial['x/d'] == ref_list1[2]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_polynomial[DF_test_polynomial['x/d'] == ref_list1[2]][[force]]))))),s=0.05, marker='D', color='cornflowerblue',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test_nn[DF_test_nn['x/d'] == ref_list2[2]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_nn[DF_test_nn['x/d'] == ref_list2[2]][[force]]))))),s=0.05, marker='D', color='lawngreen',alpha=0.5)


    plt.scatter(np.array(list(map(inverse_transY, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[3]][['y/d']]))))),
        np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[3]][[force]]))))),s=0.05, marker='D', color='darkorange',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test_polynomial[DF_test_polynomial['x/d'] == ref_list1[3]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_polynomial[DF_test_polynomial['x/d'] == ref_list1[3]][[force]]))))),s=0.05, marker='D', color='cornflowerblue',alpha=0.5)
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test_nn[DF_test_nn['x/d'] == ref_list2[3]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test_nn[DF_test_nn['x/d'] == ref_list2[3]][[force]]))))),s=0.05, marker='D', color='lawngreen',alpha=0.5)

    color = ['darkorange', 'cornflowerblue','lawngreen']
    labels = ['真实值', '多项式回归预测值','神经网络预测值']
    patches = [mpatches.Patch(facecolor=color[i], linewidth=0.02, label="{:s}".format(labels[i])) for i in
               range(len(color))]
    plt.legend(handles=patches, ncol=1, fontsize = 'x-large')
    plt.xlabel('y/d', fontsize=20)
    plt.ylabel(force, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
def pltxd_test_result(DF,df_minmax,DF_test,force):
    '''
    该函数是对横向移动的测试集数据进行图形绘制。
    :param DF:将H整体数据.csv(是对数据进行滤波)和L整体数据0414修改.csv(是对数据进行滤波)进行合并之后的数据
    :param df_minmax:将DF进行归一化之后的数据
    :param DF_test:测试集数据
    :param force:径向力还是纵向力
    :return:该函数无返回值，直接对交叉验证的预测值绘制图形
    '''
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
    ref_list1 = [pd.DataFrame(DF_test['x/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test['x/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test['x/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test['x/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[4].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[5].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[6].name]
    def inverse_transZ(x):
        max_ = DF[force].max()
        min_ = DF[force].min()
        return x * (max_ - min_) + min_
    def inverse_transX(x):
        max_ = DF['x/d'].max()
        min_ = DF['x/d'].min()
        return x * (max_ - min_) + min_
    def inverse_transY(x):
        max_ = DF['y/d'].max()
        min_ = DF['y/d'].min()
        return x * (max_ - min_) + min_
    plt.subplots(figsize=(10, 7))
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[4]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[4]][[force]]))))), s=0.05, marker='D', color='darkorange')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test[DF_test['y/d'] == ref_list1[4]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test[DF_test['y/d'] == ref_list1[4]][[force]]))))), s=0.05, marker='D', color='cornflowerblue')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[5]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[5]][[force]]))))), s=0.05, marker='D', color='darkorange')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test[DF_test['y/d'] == ref_list1[5]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test[DF_test['y/d'] == ref_list1[5]][[force]]))))), s=0.05, marker='D', color='cornflowerblue')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[6]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[6]][[force]]))))), s=0.05, marker='D', color='darkorange')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test[DF_test['y/d'] == ref_list1[6]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test[DF_test['y/d'] == ref_list1[6]][[force]]))))), s=0.05, marker='D', color='cornflowerblue')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[7]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[7]][[force]]))))), s=0.05, marker='D', color='darkorange')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test[DF_test['y/d'] == ref_list1[7]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test[DF_test['y/d'] == ref_list1[7]][[force]]))))), s=0.05, marker='D', color='cornflowerblue')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[8]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[8]][[force]]))))), s=0.05, marker='D', color='darkorange')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test[DF_test['y/d'] == ref_list1[8]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test[DF_test['y/d'] == ref_list1[8]][[force]]))))), s=0.05, marker='D', color='cornflowerblue')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[9]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[9]][[force]]))))), s=0.05, marker='D', color='darkorange')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test[DF_test['y/d'] == ref_list1[9]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test[DF_test['y/d'] == ref_list1[9]][[force]]))))), s=0.05, marker='D', color='cornflowerblue')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[10]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['y/d'] == ref_list[10]][[force]]))))), s=0.05, marker='D', color='darkorange')
    plt.scatter(np.array(list(map(inverse_transX, list(np.array(DF_test[DF_test['y/d'] == ref_list1[10]][['x/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test[DF_test['y/d'] == ref_list1[10]][[force]]))))), s=0.05, marker='D', color='cornflowerblue')


    color = ['darkorange', 'cornflowerblue']
    labels = ['真实值', '预测值']
    patches = [mpatches.Patch(facecolor=color[i], linewidth=0.02, label="{:s}".format(labels[i])) for i in
               range(len(color))]
    plt.legend(handles=patches, ncol=2, fontsize = 'x-large')
    plt.xlabel('x/d', fontsize=20)
    plt.ylabel(force, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
def pltyd_test_result(DF,df_minmax,DF_test,force):
    '''
    该函数是对纵向移动的测试集数据进行图形绘制。
    :param DF:将H整体数据.csv(是对数据进行滤波)和L整体数据0414修改.csv(是对数据进行滤波)进行合并之后的数据
    :param df_minmax:将DF进行归一化之后的数据
    :param DF_test:测试集数据
    :param force:径向力还是纵向力
    :return:该函数无返回值，直接对交叉验证的预测值绘制图形
    '''
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
    ref_list1 = [pd.DataFrame(DF_test['x/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test['x/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test['x/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test['x/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[0].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[1].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[2].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[3].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[4].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[5].name,
                 pd.DataFrame(DF_test['y/d'].value_counts()).iloc[6].name]
    def inverse_transZ(x):
        max_ = DF[force].max()
        min_ = DF[force].min()
        return x * (max_ - min_) + min_
    def inverse_transX(x):
        max_ = DF['x/d'].max()
        min_ = DF['x/d'].min()
        return x * (max_ - min_) + min_
    def inverse_transY(x):
        max_ = DF['y/d'].max()
        min_ = DF['y/d'].min()
        return x * (max_ - min_) + min_
    plt.subplots(figsize=(10, 7))
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[0]][['y/d']]))))),
        np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[0]][[force]]))))),s=0.05, marker='D', color='darkorange')
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test[DF_test['x/d'] == ref_list1[0]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test[DF_test['x/d'] == ref_list1[0]][[force]]))))),s=0.05, marker='D', color='cornflowerblue')

    plt.scatter(np.array(list(map(inverse_transY, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[1]][['y/d']]))))),
        np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[1]][[force]]))))),s=0.05, marker='D', color='darkorange')
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test[DF_test['x/d'] == ref_list1[1]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test[DF_test['x/d'] == ref_list1[1]][[force]]))))),s=0.05, marker='D', color='cornflowerblue')

    plt.scatter(np.array(list(map(inverse_transY, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[2]][['y/d']]))))),
        np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[2]][[force]]))))),s=0.05, marker='D', color='darkorange')
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test[DF_test['x/d'] == ref_list1[2]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test[DF_test['x/d'] == ref_list1[2]][[force]]))))),s=0.05, marker='D', color='cornflowerblue')

    plt.scatter(np.array(list(map(inverse_transY, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[3]][['y/d']]))))),
        np.array(list(map(inverse_transZ, list(np.array(df_minmax[df_minmax['x/d'] == ref_list[3]][[force]]))))),s=0.05, marker='D', color='darkorange')
    plt.scatter(np.array(list(map(inverse_transY, list(np.array(DF_test[DF_test['x/d'] == ref_list1[3]][['y/d']]))))),
                np.array(list(map(inverse_transZ, list(np.array(DF_test[DF_test['x/d'] == ref_list1[3]][[force]]))))),s=0.05, marker='D', color='cornflowerblue')

    color = ['darkorange', 'cornflowerblue']
    labels = ['真实值', '预测值']
    patches = [mpatches.Patch(facecolor=color[i], linewidth=0.02, label="{:s}".format(labels[i])) for i in
               range(len(color))]
    plt.legend(handles=patches, ncol=2, fontsize = 'x-large')
    plt.xlabel('y/d', fontsize=20)
    plt.ylabel(force, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
def vertical_cfd_plot(dfL,force):
    '''
    该函数是将轨迹数据进行滤波之后对数据进行可视化展示
    :param dfL: dfL指的是垂向移动的数据
    :return:无返回值
    '''
    plt.subplots(figsize=(8, 6))
    plt.plot(dfL[dfL['x/d'] == 0].iloc[:1110]['y/d'], dfL[dfL['x/d'] == 0].iloc[:1110][force], label=r'x/d=0',
             linestyle='dashed', color='cornflowerblue')
    plt.plot(dfL[dfL['x/d'] == 0].iloc[1110:]['y/d'], dfL[dfL['x/d'] == 0].iloc[1110:][force], linestyle='dashed',
             color='cornflowerblue')
    plt.plot(dfL[dfL['x/d'] == 0.5].iloc[:3205]['y/d'], dfL[dfL['x/d'] == 0.5].iloc[:3205][force], linestyle='-.',
             color='darkorange')
    plt.plot(dfL[dfL['x/d'] == 0.5].iloc[3205:]['y/d'], dfL[dfL['x/d'] == 0.5].iloc[3205:][force], label=r'x/d=0.5',
             linestyle='-.', color='darkorange')
    plt.plot(dfL[dfL['x/d'] == 1.5].iloc[:1691]['y/d'], dfL[dfL['x/d'] == 1.5].iloc[:1691][force], label=r'x/d=1.5',
             linestyle=':', color='red')
    plt.plot(dfL[dfL['x/d'] == 1.5].iloc[1692:3006]['y/d'], dfL[dfL['x/d'] == 1.5].iloc[1692:3006][force],
             linestyle=':', color='red')
    plt.plot(dfL[dfL['x/d'] == 1.5].iloc[3007:]['y/d'], dfL[dfL['x/d'] == 1.5].iloc[3007:][force], linestyle=':',
             color='red')
    plt.plot(dfL[dfL['x/d'] == 2].iloc[:3068]['y/d'], dfL[dfL['x/d'] == 2].iloc[:3068][force], label=r'x/d=2.0',
             linestyle=':', color='magenta')
    plt.plot(dfL[dfL['x/d'] == 2].iloc[3069:]['y/d'], dfL[dfL['x/d'] == 2].iloc[3069:][force], linestyle=':',
             color='magenta')
    plt.xlabel('y/d', size=20)
    if force=='CFv':
        plt.ylabel('Cfv', size=20)
    elif force=='CFr':
        plt.ylabel('Cfv', size=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize='x-large')
    plt.show()
def transverse_cfd_plot(dfH,force):
    '''
    该函数是将轨迹数据进行滤波之后对数据进行可视化展示
    :param dfL: dfH指的是横向移动的数据
    :return:无返回值
    '''
    plt.subplots(figsize=(8, 9))
    plt.plot(dfH[dfH['y/d'] == 0.750]['x/d'], dfH[dfH['y/d'] == 0.750][force], label=r'y/d=0.750', linestyle='dashed')
    plt.plot(dfH[dfH['y/d'] == 0.875]['x/d'], dfH[dfH['y/d'] == 0.875][force], label=r'y/d=0.875', linestyle='-.')
    plt.plot(dfH[dfH['y/d'] == 1.000]['x/d'], dfH[dfH['y/d'] == 1.000][force], label=r'y/d=1.000', linestyle=':')
    plt.plot(dfH[dfH['y/d'] == 1.125]['x/d'], dfH[dfH['y/d'] == 1.125][force], label=r'y/d=1.125', linestyle='-')
    plt.plot(dfH[dfH['y/d'] == 1.250]['x/d'], dfH[dfH['y/d'] == 1.250][force], label=r'y/d=1.250',
             linestyle=(0, (3, 1, 1, 1)))
    plt.plot(dfH[dfH['y/d'] == 1.375]['x/d'], dfH[dfH['y/d'] == 1.375][force], label=r'y/d=1.375',
             linestyle=(0, (3, 5, 1, 5)))
    plt.plot(dfH[dfH['y/d'] == 1.500]['x/d'], dfH[dfH['y/d'] == 1.500][force], label=r'y/d=1.500',
             linestyle=(0, (3, 10, 1, 10, 1, 10)))
    plt.xlabel('x/d', fontsize=20)
    if force=='CFv':
        plt.ylabel('Cfv', size=20)
    elif force=='CFr':
        plt.ylabel('Cfv', size=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
def h55fv_contrast_cfd_test():
    '''
    将h55fv的cfd数据和test数据进行对比绘制图形
    '''
    df = pd.read_excel(r'D:\WORK_DATA_F\suction of mineral\h55_fv.xlsx')
    CFD = df[['CFDx(mm)', 'CFDFv(N)']]
    CFD = CFD.dropna(axis=0, how='all')
    CFD = CFD[(CFD['CFDx(mm)'] >= 0) & (CFD['CFDx(mm)'] <= 80)]
    CFD = CFD.sort_index(ascending=False)
    CFD.reset_index(drop=True, inplace=True)
    test = df[['testx(mm)', 'testFv(N)']]
    test = test[(test['testx(mm)'] >= 0) & (test['testx(mm)'] <= 80)]
    test.reset_index(drop=True, inplace=True)
    datacfd=pywt_calculte((np.array((CFD['CFDFv(N)']))),'db8',8)
    tran_CFD=pd.DataFrame([np.array(CFD['CFDx(mm)']),datacfd[:-1]],index=['tran_CFDx(mm)','tran_CFDFv(N)']).T
    datatest=pywt_calculte((np.array((test['testFv(N)']))),'db8',11)
    tran_test=pd.DataFrame([np.array(test['testx(mm)']),datatest[:-1]],index=['tran_testx(mm)','tran_testFv(N)']).T
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体为新宋体。
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots(figsize=(8, 6))
    plt.plot(tran_CFD['tran_CFDx(mm)'], tran_CFD['tran_CFDFv(N)'], color='r', label='滤波后的仿真数据')
    plt.plot(CFD['CFDx(mm)'], CFD['CFDFv(N)'], alpha=0.2, label='仿真数据')
    plt.plot(test['testx(mm)'], test['testFv(N)'], alpha=0.5, label='试验数据')
    plt.plot(tran_test['tran_testx(mm)'], tran_test['tran_testFv(N)'], alpha=0.5, label='滤波后的试验数据')
    plt.xlabel('横向距离(mm)', size=20)
    plt.ylabel('Fv(N)', size=20)
    plt.legend(fontsize='x-large')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
def h55fr_contrast_cfd_test():
    '''
    将h55fv的cfd数据和test数据进行对比绘制图形
    '''
    df = pd.read_excel(r'D:\WORK_DATA_F\suction of mineral\h55_fr.xlsx')
    CFD = df[['CFDx(mm)', 'CFDFr(N)']]
    CFD = CFD.dropna(axis=0, how='all')
    CFD = CFD[(CFD['CFDx(mm)'] >= 0) & (CFD['CFDx(mm)'] <= 80)]
    CFD = CFD.sort_index(ascending=False)
    CFD.reset_index(drop=True, inplace=True)
    test = df[['testx(mm)', 'testFr(N)']]
    test = test[(test['testx(mm)'] >= 0) & (test['testx(mm)'] <= 80)]
    test.reset_index(drop=True, inplace=True)
    datacfd=pywt_calculte((np.array((CFD['CFDFr(N)']))),'db8',8)
    tran_CFD=pd.DataFrame([np.array(CFD['CFDx(mm)']),datacfd[:-1]],index=['tran_CFDx(mm)','tran_CFDFr(N)']).T
    datatest=pywt_calculte((np.array((test['testFr(N)']))),'db8',11)
    tran_test=pd.DataFrame([np.array(test['testx(mm)']),datatest[:-1]],index=['tran_testx(mm)','tran_testFr(N)']).T
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体为新宋体。
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots(figsize=(8, 6))
    plt.plot(tran_CFD['tran_CFDx(mm)'], -tran_CFD['tran_CFDFr(N)'], color='r', label='滤波后的仿真数据')
    plt.plot(CFD['CFDx(mm)'], -CFD['CFDFr(N)'], alpha=0.2, label='仿真数据')
    plt.plot(test['testx(mm)'], -test['testFr(N)'], alpha=0.5, label='试验数据')
    plt.plot(tran_test['tran_testx(mm)'], -tran_test['tran_testFr(N)'], alpha=0.5, label='滤波后的试验数据')
    plt.xlabel('横向距离(mm)', size=20)
    plt.ylabel('Fr(N)', size=20)
    plt.legend(fontsize='x-large')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
def surface_domain_poly(DF,df_minmax,force,machine_model):
    '''
    :param DF: 指的是将H整体数据.csv(是对数据进行滤波)和L整体数据0414修改.csv(是对数据进行滤波)进行合并之后的数据
    :param df_minmax: 指的是将DF进行归一化之后的数据
    :param force: 指的是径向力还是纵向力
    :param machine_model: 指的是人工神经网络模型还是多项式回归模型
    :return: 无返回值
    '''
    n = 600
    # 做点
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    # 构造点
    X, Y = np.meshgrid(x, y)
    z = [i for i in zip(X.flat, Y.flat)]
    X_data = np.array(df_minmax[['x/d', 'y/d']])
    y_data = np.array(df_minmax[[force]]).flatten()
    X_data = X_data.astype(np.float32)
    y_data = y_data.astype(np.float32)
    if machine_model=='polynomial':
        model=polynomial_model_built()
        model.fit(X_data, y_data)
        Z = model.predict(np.array(z))
    elif machine_model=='nn':
        if force=='CFv':
            model=nn_model_built_Fv()
        elif force=='CFr':
            model=nn_model_built_Fr()
        history = model.fit(X_data, y_data,
                                  epochs=30,
                                  batch_size=256,
                                  verbose=0,
                                  shuffle=True)
        Z = model.predict(np.array(z))
    else:
        print('模型输入错误')
    def inverse_transZ(x):
        max_ = DF[force].max()
        min_ = DF[force].min()
        return x * (max_ - min_) + min_
    def inverse_transX(x):
        max_ = DF['x/d'].max()
        min_ = DF['x/d'].min()
        return x * (max_ - min_) + min_
    def inverse_transY(x):
        max_ = DF['y/d'].max()
        min_ = DF['y/d'].min()
        return x * (max_ - min_) + min_
    X_ = np.array(list(map(inverse_transX, list(X))))
    Y_ = np.array(list(map(inverse_transY, list(Y))))
    Z_ = np.array(list(map(inverse_transZ, list(Z))))
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax0 = ax.pcolormesh(X_, Y_, Z_.reshape(600, 600), shading='auto')
    if force=='CFv':
        C = plt.contour(X_, Y_, Z_.reshape(600, 600), [0.1, 0.1438,0.2, 0.3, 0.4, 0.5, 0.6, 0.7], extend='max', cmap='summer')
    elif force=='CFr':
        C = plt.contour(X_, Y_, Z_.reshape(600, 600), [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], extend='max', cmap='summer')
    ax.clabel(C, inline=True, colors='k', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.colorbar(ax0,fontsize=20)
    plt.xlabel('x/d',fontsize=20)
    plt.ylabel('y/d',fontsize=20)
    if force=='CFv':
        plt.title('Cfv',fontsize=20)
    elif force=='CFr':
        plt.title('Cfr',fontsize=20)
    # plt.savefig('多项式回归径向力.jpg')
    plt.show()




