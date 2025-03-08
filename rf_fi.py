# 导入数据库
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
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
X = df_csv.iloc[:, 0:63]  #1-63传感器作为输入特征
Y = df_csv.iloc[:, 63:66]  #64-66负载率、故障高度、故障电流比作为输出标签

# 使用前6行作为训练集，后3行作为测试集
X_train, X_test = X.iloc[0:6, :], X.iloc[6:9, :]
Y_train, Y_test = Y.iloc[0:6, :], Y.iloc[6:9, :]


# 适应度函数
def evaluate(individual):
    # 传感器选择
    selected_features = [i for i in range(individual) if individual[i] == 1]

    if len(selected_features) == 0:
        return 0,

    #查看选中的传感器
    print("Selected features:", selected_features)

    # 用选定的传感器数据作为特征
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    Y_train_selected = Y_train.iloc[:, selected_features]
    Y_test_selected = Y_test.iloc[:, selected_features]

    # 使用随机森林训练模型
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_selected, Y_train_selected)

    # 进行预测
    y_pred = model.predict(X_test_selected)

    # 计算MSE或准确度作为适应度
    mse = mean_squared_error(Y_test_selected, y_pred)
    accuracy = 1 / (1 + mse)

    # 惩罚因素：传感器越少，惩罚越大
    # 需要仔细考虑惩罚项！！！！
    penalty = len(selected_features) / len(X.columns)
    # 精度大于0.9时，停止惩罚
    if accuracy > 0.9:
        return accuracy - penalty,
    else:
        return accuracy - penalty,


# 设置遗传算法的创造者
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化适应度
creator.create("Individual", list, fitness=creator.FitnessMax)


# 遗传算法的个体创建器
def create_individual():
    return [np.random.randint(0, 1) for _ in range(X_train.shape[1])]  # 随机选择0或1，表示传感器是否被选中


# 设置遗传算法中的工具
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)  # 交叉操作
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)  # 变异操作
toolbox.register("select", tools.selTournament, tournsize=3)  # 选择操作
toolbox.register("evaluate", evaluate)

# 初始化种群
population = toolbox.population(n=50)

# 遗传算法的参数
generations = 50
cx_prob = 0.7  # 交叉概率
mut_prob = 0.2  # 变异概率

# 执行遗传算法
algorithms.eaSimple(population, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=generations,
                    stats=None, halloffame=None, verbose=True)

# 获取最优解
best_individual = tools.selBest(population, 1)[0]
print("最优传感器组合为：", best_individual)

# 输出选定传感器的数量和性能
selected_features = [i for i in range(len(best_individual)) if best_individual[i] == 1]
print(f"选定的传感器数量: {len(selected_features)}")

# 用最优传感器组合训练并测试模型
X_train_selected = X_train.iloc[:, selected_features]
X_test_selected = X_test.iloc[:, selected_features]
Y_train_selected = Y_train.iloc[:, selected_features]
Y_test_selected = Y_test.iloc[:, selected_features]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_selected, Y_train_selected)
y_pred = model.predict(X_test_selected)

# 输出模型的MSE
mse = mean_squared_error(Y_test, y_pred)
print(f"最优传感器组合的均方误差: {mse:.2f}")

# 提取特征重要性
feature_importances = model.feature_importances_
selected_feature_names = [f"Sensor {i + 1}" for i in selected_features]  # 提取传感器名称

# 将特征重要性和名称放入DataFrame
importance_df = pd.DataFrame({
    'Feature Importance': feature_importances,
    'Features': selected_feature_names
})

# 使用seaborn绘制条形图
plt.figure(figsize=(10, 6))
sns.barplot(x='Feature Importance', y='Features', data=df)
plt.title('Feature Importance of Sensors')
plt.show()
