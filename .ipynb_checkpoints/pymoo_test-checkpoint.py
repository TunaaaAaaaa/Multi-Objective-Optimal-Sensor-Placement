#该文件用于pymoo的多目标优化代码测试及优化练习

#导入数据库
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 将excel文件转成csv文件
excelfile = r'E:\\FaultDetec_SensorOpti\\python\\pythonProject\\training_sample.xlsx'
csvfile = r'E:\\FaultDetec_SensorOpti\\python\\pythonProject\\training_sample.csv'
df = pd.read_excel(excelfile)
df.to_csv(csvfile, index=False)

# 导入数据集
df_csv = pd.read_csv(csvfile)

# 分割数据集
Input = df_csv.iloc[:, 0:63]  #1-63传感器作为输入特征
Output = df_csv.iloc[:, 63:66]  #64-66负载率、故障高度、故障电流比作为输出标签

# 使用前6行作为训练集，后3行作为测试集
Input_train, Input_test = Input.iloc[0:6, :], Input.iloc[6:9, :]
Output_train, Output_test = Output.iloc[0:6, :], Output.iloc[6:9, :]

#############导入数据集结束#################

#导入问题定义包
from pymoo.core.problem import Problem

class MyProblem(Problem):
    def _init_(self):
        super()._init_(n_var=63, 
                       n_obj=2, 
                       n_constr=0,
                       xl=0, xu=1, 
                       vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        #从x中提取传感器序号以及数量
        selected_sensors = [i for i in range(len(x)) if x[i] == 1]
        sensors_num = len(selected_sensors)
        print("selected_sensors:", selected_sensors)
        print("selected_sensors_number:", sensors_num)

        # 用选定的传感器数据作为特征
        Input_train_selected = Input_train.iloc[:, selected_sensors]
        Input_test_selected = Input_test.iloc[:, selected_sensors]

        # 使用随机森林训练模型
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42) #使用RandomForestRegressor作为分类器
        model.fit(Input_train_selected, Output_train) #使用选中的Input和Output进行分类器训练
        output_pred = model.predict(Input_test_selected) #使用选中的测试集输入预测输出
        mse = mean_squared_error(Output_test, output_pred) #使用均方差计算输出的预测值和实际值的误差

        #定义目标函数
        f1 = len(selected_sensors)
        f2 = mse
        out["F"] = [f1, f2]

problem = MyProblem()

#导入算法包
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

algorithm = NSGA2(
    pop_size = 40,
    sampling = BinaryRandomSampling(),
    crossover = TwoPointCrossover(),
    mutation = BitflipMutation(),
    eliminate_duplicates = True)

res = minimize(problem,
               algorithm,
               ('n_gen',100),
               verbose = False)

print("Best solution found: %s" % res.X.astype(int))
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)