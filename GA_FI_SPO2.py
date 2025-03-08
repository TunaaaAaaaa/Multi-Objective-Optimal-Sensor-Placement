# 导入数据库
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling  # 用于生成0或1
from pymoo.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pymoo.visualization.scatter import Scatter
import random
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 将excel文件转成csv文件
excelfile = r'E:\\FaultDetec_SensorOpti\\python\\pythonProject\\training_sample.xlsx'
csvfile = r'E:\\FaultDetec_SensorOpti\\python\\pythonProject\\training_sample.csv'
df = pd.read_excel(excelfile)
df.to_csv(csvfile, index=False)
# # 将最新建立的数据集替换掉愿来的数据集
# csvfile = r'E:\\FaultDetec_SensorOpti\\dataset\\normal_fault_data.csv'

# 导入数据集
df_csv = pd.read_csv(csvfile)
test_size = 0.3

# 分割数据集
Input = df_csv.iloc[:, 0:63]  #1-63传感器作为输入特征
Output = df_csv.iloc[:, 63:66]  #64-66负载率、故障高度、故障电流比作为输出标签
# # 分割新的数据集
# Input = df_csv.iloc[:, 1:64]  # 除第一列为文件名外，第2到65列作为输入特征
# Output = df_csv.iloc[:, -1]  # 最后一列作为输出标签，0为正常，1为故障

# 使用前6行作为训练集，后3行作为测试集
Input_train, Input_test = Input.iloc[0:6, :], Input.iloc[6:9, :]
Output_train, Output_test = Output.iloc[0:6, :], Output.iloc[6:9, :]

# # 使用train_test_split来分割新的训练集和数据集
# Input_train, Input_test = train_test_split(Input, test_size=test_size, random_state=42)
# Output_train, Output_test = train_test_split(Output, test_size=test_size, random_state=42)
# print('Input_train is:', Input_train)
# print('Output_train is:', Output_train)
# print('Input_test_train is:', Input_test)
# print('Output_test_train is:', Output_test)
#############导入数据集结束#################



# 导入问题定义包
from pymoo.core.problem import Problem


class MyProblem(Problem):
    def __init__(self, Input_train, Output_train, Input_test, Output_test):

        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = 0 * np.ones(63)
        xu = 1 * np.ones(63)

        super().__init__(n_var=63,
                         n_obj=2,
                         n_constr=0,
                         xl=xl, xu=xu,
                         vtype=int)

        # 保存数据
        self.Input_train = Input_train
        self.Output_train = Output_train
        self.Input_test = Input_test
        self.Output_test = Output_test

    def _evaluate(self, x, out, *args, **kwargs):
        # x为种群,每一行为一个个体(染色体)

        # print("x is:", x)
        # print("x size is:", len(x), len(x[0]))

        # 初始化存储 f1 和 f2 值的列表
        f1_values = []
        f2_values = []

        # 遍历种群中的每个个体（每一行是一个染色体）
        for i in range(x.shape[0]):  # 遍历种群中的每一个个体
            individual = x[i]  # 选取当前个体（每一行）

            # 从individual中提取传感器序号以及数量

            selected_sensors = [index for index in range(len(individual)) if individual[index] == 1]
            sensors_num = len(selected_sensors)

            if sensors_num == 0:
                mse = float('inf')  # 如果没有选择任何传感器，定义均方误差为无限大
            else:
                print("selected_sensors:", selected_sensors)
                print("selected_sensors_number:", sensors_num)

                # 用选定的传感器数据作为特征
                Input_train_selected = Input_train.iloc[:, selected_sensors]
                Input_test_selected = Input_test.iloc[:, selected_sensors]

                # 使用随机森林训练模型
                model = RandomForestRegressor(n_estimators=100, max_depth=5,
                                               random_state=42)  # 使用RandomForestClassifier作为分类器
                model.fit(Input_train_selected, Output_train)  # 使用选中的Input和Output进行分类器训练
                output_pred = model.predict(Input_test_selected)  # 使用选中的测试集输入预测输出
                mse = mean_squared_error(Output_test, output_pred)  # 使用均方差计算输出的预测值和实际值的误差

            # 定义目标函数
            f1 = sensors_num
            f2 = mse

            f1_values.append(f1)
            f2_values.append(f2)

            print("sensors number is:", f1_values, "mse is:", f2_values)

        # 将目标函数值传递给 out
        out["F"] = np.column_stack([f1_values, f2_values])



problem = MyProblem(Input_train, Output_train, Input_test, Output_test)
print("n of var is:", problem.n_var)
print(problem.n_obj)
# algorithm : NSGA2, reference in NSGA-II: Non-dominated Sorting Genetic Algorithm  in pymoo

algorithm = NSGA2(pop_size=200,
                  sampling=IntegerRandomSampling(),
                  crossover=TwoPointCrossover(),
                  mutation=BitflipMutation(),
                  eliminate_duplicate=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 5),
               seed=1,
               verbose=True)


# 设置字体
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 绘制散点图并传入求解结果
scatter = Scatter()
scatter.add(res.F)
scatter.show()



print(res.F)
print(res.X)