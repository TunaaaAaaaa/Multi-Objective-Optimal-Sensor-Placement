#整合csv数据文件，并标注工况
import pandas as pd
import os

# #先将故障工况数据合并：
# #指定ascii文件所在文件夹
# faultdata_path = "E:\\FaultDetec_SensorOpti\\dataset\\faultdata"
# faultdata_output = "faultdata.csv"
#
# #获取所有ascii文件
# ascii_files = [f for f in os.listdir(faultdata_path)]
#
# all_data = []
#
# for file in ascii_files:
#     file_path = os.path.join(faultdata_path, file)
#     df = pd.read_table(file_path, delimiter = ',', engine = 'python') #读取逗号分割的文件
#
#     fifth_column_data = df.iloc[1:, 4].values #提取第五列除第一行外的数据
#     row_data = [file] + list(fifth_column_data) #添加文件名并转为行向量
#     all_data.append(row_data)
#
# # 保存数据
# output_df = pd.DataFrame(all_data)
# output_df.to_csv(os.path.join(faultdata_path, faultdata_output), index = False, header = False)
#
# print(f"合并文件已保存为 {faultdata_output}")


#再将正常工况下的数据合并为normaldata，储存在normaldata文件夹中
#先将正常工况数据合并：
#指定ascii文件所在文件夹
normaldata_path = "E:\\FaultDetec_SensorOpti\\dataset\\normaldata"
normaldata_output = "normaldata.csv"

#获取所有csv文件
ascii_files = [f for f in os.listdir(normaldata_path)]

all_data = []

for file in ascii_files:
    file_path = os.path.join(normaldata_path, file)
    df = pd.read_table(file_path, delimiter = ',', engine = 'python') #读取逗号分割的文件

    fifth_column_data = df.iloc[1:, 4].values #提取第五列除第一行外的数据
    row_data = [file] + list(fifth_column_data) #添加文件名并转为行向量
    all_data.append(row_data)

# 保存数据
output_df = pd.DataFrame(all_data)
output_df.to_csv(os.path.join(normaldata_path, normaldata_output), index = False, header = False)

print(f"合并文件已保存为 {normaldata_output}")