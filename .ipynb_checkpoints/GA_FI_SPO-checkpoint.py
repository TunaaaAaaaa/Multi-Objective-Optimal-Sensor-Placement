# 导入数据库
import random
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
def evaluate(chromosome):
    # 传感器选择
    # print(chromosome)
    selected_features = [i for i in range(len(chromosome)) if chromosome[i] == 1]

    # print(selected_features)

    if len(selected_features) == 0:
        return 0,

    #查看选中的传感器
    print("Selected features:", selected_features)

    # 用选定的传感器数据作为特征
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]


    # 使用随机森林训练模型
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_selected, Y_train)

    # 进行预测
    y_pred = model.predict(X_test_selected)

    # 计算MSE或准确度作为适应度
    mse = mean_squared_error(Y_test, y_pred)
    accuracy = 1 / (1 + mse)

    # 惩罚因素：传感器越多，惩罚越大
    penalty_weight = 0.1  # 惩罚权重
    penalty = penalty_weight * len(selected_features) / len(X.columns)

    return accuracy - penalty


# 初始化种群
def initialize_population(population_size, bounds, chromosome_size):
    population = []
    for _ in range(population_size):
        #个体的每个值在0或1之间随机取整
        chromosome = np.random.randint(bounds[0],bounds[1],size = chromosome_size).tolist()
        population.append(chromosome)

    return population

# # 初始化染色体
# def initialize_chromosome(chromosome_size, bounds):
#     chromosome = []
#     for _ in range(chromosome_size):
#         #染色体的每一个元素在0，1之间随机取整
#         unit = np.random.randint(bounds[0],bounds[1])
#         chromosome.append(unit)
#
#     return chromosome


# 选择（轮盘赌选择） 不是很好用，先不用了
# def select(population, fitness):
#     min_fitness = min(fitness)
#     if min_fitness < 0:
#         # 平移使所有适应度非负
#         fitness = [f - min_fitness for f in fitness]
#     total_fitness = sum(fitness) + 1e-3
#     probabilities = [f / total_fitness for f in fitness]
#     selected = np.random.choice(population, size=len(population), p=probabilities)
#     print("length of population:", len(population))
#     print("length of p:", len(probabilities))
#     return selected

# 锦标赛选择
def tournament_select(population, population_size, fitness, tournament_size):
    new_population = []
    new_fitness = []
    # while循环指导新的种群规模达到当前种群规模
    while len(new_population) < population_size:
        # 从原始样本中选出新的样本
        # checked_list_population = [random.randint(0, population_size) for i in range(0, tournament_size + 1)]
        checked_list_population = random.sample(range(population_size), tournament_size)
        # 样本对应的适应度
        checked_list_fitness = np.array([fitness[i] for i in checked_list_population])
        # checked_list_fitness = [fitness[i] for i in checked_list_population]

        # # 选中样本的最大适应度
        # max_fitness = checked_list_fitness.max()
        # # 最大适应度样本对应的索引
        # idx = np.where(checked_list_population == max_fitness)[0][0]
        # # 最大适应度样本对应的样本
        # max_population = population[idx]

        # 找到适应度最大个体的索引
        idx = np.argmax(checked_list_fitness)
        # 选出具有最大适应度的个体
        max_population = population[checked_list_population[idx]]
        # 放入新的种群和适应度中
        new_population.append(max_population)

        # new_fitness.append(max_fitness)

    # return new_population, new_fitness
        return new_population

# 交叉（单点交叉） （chromosomes）
def crossover(parent1, parent2):
    # 确保parent1和parent2是二进制编码（0或1的数组）
    crossover_point = random.randint(1, len(parent1) - 1)  # 随机选择交叉点
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2


# 变异
def mutate(chromosome, mutation_rate):
    # 对每个基因位进行独立变异
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:  # 以一定的变异率进行变异
            chromosome[i] = 1 - chromosome[i]  # 0变1，1变0
    return chromosome


# 遗传算法主函数
def genetic_algorithm(bounds, population_size, generations, mutation_rate, chromosome_size):
    # 初始化种群
    population = initialize_population(population_size, bounds, chromosome_size)
    best_fitness_over_time = []

    for generation in range(generations):
        # 评估种群适应度
        fitness = evaluate(population)
        best_fitness_over_time.append(max(fitness))

        # 选择
        selected_population = tournament_select(population, population_size, fitness, tournament_size)


        # 生成新种群
        new_population = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = new_population

    # 返回最佳个体及其适应度随时间变化
    best_individual = max(population, key=evaluate)
    return best_individual, best_fitness_over_time

# 参数设置
bounds = [0, 2]
population_size = 5
chromosome_size = 63
tournament_size = 2
generations = 50
mutation_rate = 0.1

# 执行遗传算法
# best_solution, fitness_over_time = genetic_algorithm(bounds, population_size, generations, mutation_rate, chromosome_size)
##########################################
population = initialize_population(population_size, bounds, chromosome_size)
best_fitness_over_time = []

for generation in range(generations):
    # 评估种群适应度
    print(population)
    fitness=np.zeros((len(population)),dtype=float)
    for i in range(len(population)):
        population_sub=population[i]
        fitness[i] = evaluate(population_sub)
    print('signal')
    best_fitness_over_time.append(max(fitness))

    # 选择
    # selected_population = tournament_select(population, population_size, fitness, tournament_size)
    ########################################
    new_population = []
    new_fitness = []
    # while循环指导新的种群规模达到当前种群规模
    while len(new_population) < population_size:
        # 从原始样本中选出新的样本
        # checked_list_population = [random.randint(0, population_size) for i in range(0, tournament_size + 1)]
        checked_list_population = random.sample(range(population_size), tournament_size)
        print(checked_list_population)
        print(fitness)
        # 样本对应的适应度
        checked_list_fitness = np.array([fitness[i] for i in checked_list_population])
        # checked_list_fitness = [fitness[i] for i in checked_list_population]

        # # 选中样本的最大适应度
        # max_fitness = checked_list_fitness.max()
        # # 最大适应度样本对应的索引
        # idx = np.where(checked_list_population == max_fitness)[0][0]
        # # 最大适应度样本对应的样本
        # max_population = population[idx]

        # 找到适应度最大个体的索引
        idx = np.argmax(checked_list_fitness)
        # 选出具有最大适应度的个体
        max_population = population[checked_list_population[idx]]
        # 放入新的种群和适应度中
        new_population.append(max_population)
        selected_population=new_population
    ##########################################

    # 生成新种群
    new_population = []
    for i in range(0, len(selected_population), 2):
        parent1 = selected_population[i]
        if i==len(population)-1:
            parent2 = selected_population[0] #改成随机数
        else:
            print('i:',i)
            print(len(population))
            parent2 = selected_population[i + 1]
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1, mutation_rate))
        new_population.append(mutate(child2, mutation_rate))

    population = new_population

# 返回最佳个体及其适应度随时间变化
best_individual = max(population, key=evaluate)
################################################################

# 输出最佳解
best_solution_value = evaluate(best_solution)

# 绘制适应度变化图
plt.plot(fitness_over_time)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Genetic Algorithm Optimization')
plt.show()

best_solution, best_solution_value
