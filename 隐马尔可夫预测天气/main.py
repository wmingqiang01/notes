import numpy as np
from HMM import HMM

# 随机生成二十天的天气数据
weather_data = np.array([3, 1, 1, 2, 1, 0, 2, 0, 3, 2, 3, 3, 1, 3, 3, 0, 3, 3, 3, 3])
print(weather_data)
# 初始化初始隐状态概率
startprob = np.array([0.25, 0.25, 0.25,0.25])

# 初始化状态转移矩阵
transmat = np.array([[0.25, 0.25, 0.25, 0.3],
                     [0.0, 0.3,0.35, 0.35],
                     [0.2, 0.4, 0.25, 0.2],
                     [0.3, 0.2, 0.25,0.25]])
observation = np.array([[0.25, 0.25, 0.25, 0.3],
                     [0, 0.5,0.25, 0.25],
                     [0, 0.4, 0.45, 0.2],
                     [0.3, 0.2, 0.25,0.25]])
# 构造 HMM 模型
model = HMM(num_states=4, num_obs=4, pi=startprob, A=transmat, B=observation)

# 使用 Baum-Welch 算法对模型参数进行训练
model.baum_welch_train(weather_data)

# 预测未来五天的天气情况
future_data = np.array([1, 1, 1, 2, 3]) # 未来 5 天都是晴天
predicted_weather = model.viterbi_predict(future_data)

# 将状态转换成具体的天气情况
weather_dict = {0: "晴天", 1: "雨天", 2:'阴天', 3:'多云'}
predicted_weather_text = [weather_dict[status] for status in predicted_weather]

# 输出预测结果
print("未来五天的天气情况预测为：", predicted_weather_text)
