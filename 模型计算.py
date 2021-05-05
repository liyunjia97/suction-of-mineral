# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from training_model import training_11_fold
from plot_result import pltxd_test_result,pltyd_test_result,surface_domain_poly,pltxd_test_result_contrast,pltyd_test_result_contrast
from read_data import read_data
plt.rcParams['font.sans-serif'] = ['SimSun']    # 指定默认字体为新宋体。
plt.rcParams['axes.unicode_minus'] = False      # 解决保存图像时 负号'-' 显示为方块和报错的问题。
def train_result_plot(force,machine_model):
    # 该函数是对数据进行训练，得到其11折交叉验证的预测结果。
    # 返回各个指标及纵向移动和径向移动的预测受力和cfd受力的对比并返回面域图的绘制结果
    DF_test,mse_data_mean,mae_data_mean,me_data_mean,mse_test_mean,mae_test_mean,me_test_mean=training_11_fold(DF,df_minmax,force,machine_model)
    print('训练集MAX_ERROR:',me_data_mean)
    print('测试集MAX_ERROR:',me_test_mean)
    print('训练集MSE:', mse_data_mean)
    print('测试集MSE:', mse_test_mean)
    print('训练集MAE:',mae_data_mean)
    print('测试集MAE:', mae_test_mean)
    pltxd_test_result(DF,df_minmax,DF_test,force)
    pltyd_test_result(DF,df_minmax,DF_test,force)
    surface_domain_poly(DF, df_minmax, force,machine_model)
def train_result_contras_plot(force):
    # 该函数是对数据进行训练，得到其11折交叉验证的预测结果。
    # 返回各个指标及纵向移动和径向移动的在不同模型下预测受力和cfd受力的对比
    DF_test_polynomial, mse_data_mean_polynomial, mae_data_mean_polynomial, me_data_mean_polynomial, mse_test_mean_polynomial, mae_test_mean_polynomial, me_test_mean_polynomial = training_11_fold(
        DF, df_minmax, force, 'polynomial')
    DF_test_nn, mse_data_mean_nn, mae_data_mean_nn, me_data_mean_nn, mse_test_mean_nn, mae_test_mean_nn, me_test_mean_nn = training_11_fold(
        DF, df_minmax, force, 'nn')
    pltxd_test_result_contrast(DF, df_minmax, DF_test_polynomial, DF_test_nn, force)
    pltyd_test_result_contrast(DF, df_minmax, DF_test_polynomial, DF_test_nn, force)
if __name__=='__main__':
    df_minmax, DF, dfL, dfH = read_data()
    train_result_plot('CFv','nn') #这个函数可直接更改为nn或者polynomial，可直接更改CFv和CFr
    train_result_contras_plot('CFr')#这个函数可直接更改是CFv还是CFr

