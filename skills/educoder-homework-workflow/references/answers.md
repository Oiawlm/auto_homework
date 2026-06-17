# 人工智能作业答案记录

更新时间：2026-06-17

说明：本文件只记录已经在平台页面中填写或通过评测的答案，便于在同类账号中复用。账号、密码不写入此文件。

## 基于大模型的提示词设计实验

任务页：

```text
http://10.210.0.247/tasks/i7gwoujycph3
```

状态说明：该任务页面测试说明写明“无需通关成功”。当前填写内容如下：

```text
OpenAI：GPT-4、ChatGPT
Google：Gemini
百度：文心一言
阿里巴巴：通义千问
智谱AI：ChatGLM
DeepSeek：DeepSeek-R1
```

## 大模型在深度学习中的算法 - 第3关：Transformer

任务页：

```text
http://10.210.0.247/tasks/nqh6eymf753p
```

评测结果：`1/1 全部通过`

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义Transformer模型类
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        ############### Begin ###############
        # 定义词嵌入层，将输入的词汇表索引转换为指定维度的向量表示
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        # 定义位置编码，用于保留序列的位置信息
        self.position_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))

        # 定义Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)

        # 定义Transformer编码器，由多个编码器层堆叠而成
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 定义全连接层，将编码器的输出映射到指定维度的输出
        self.fc = nn.Linear(hidden_dim, output_dim)
        ############### End ###############

    def forward(self, x):
        ############### Begin ###############
        # 将输入词汇表索引转换为向量表示，并加上位置编码
        x = self.embedding(x) + self.position_encoding[:, :x.size(1), :]

        # 将输入传入编码器进行处理
        x = self.encoder(x)

        # 对编码器的输出取平均，作为全连接层的输入
        x = self.fc(x.mean(dim=1))
        return x
        ############### End ###############

# 参数设定
input_dim = 10000  # 输入词汇表大小，即词汇表中不同词汇的数量
hidden_dim = 512   # 隐藏层大小，即词嵌入和编码器的输出维度
output_dim = 10    # 输出类别数量，即分类任务的类别数量
num_heads = 8      # 多头注意力机制的头数
num_layers = 6     # 编码器层的数量

# 实例化模型
model = TransformerModel(input_dim, hidden_dim, output_dim, num_heads, num_layers)

# 查看模型结构
print(model)
```

## 回归模型效果评估

任务页：

```text
http://10.210.0.247/tasks/in4ov8fs6qrb
```

评测结果：`1/1 全部通过`

注意：该关参考答案中 `mean_absolute_percentage_error` 不乘以 100，`max_error` 直接传入二维数组。

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error, mean_absolute_percentage_error

# 示例数据生成
np.random.seed(0)  # 设置随机种子以保证结果可复现
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 模型评估
########## Begin ##########
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
max_err = max_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
########## End ##########

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R^2 Score: {r2:.2f}')
print(f'Max Error: {max_err:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
```

## 梯度下降算法

任务页：

```text
http://10.210.0.247/tasks/n92qb74fhzfx
```

评测结果：`2/2 全部通过`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
##########Begin##########
def grad_1d(x,b):
    """
    目标函数的梯度
    :param x: 自变量，标量
    :return: 因变量，标量
    """
    return x * 2+b

def gradient_descent_1d(grad,parameter):
    """
    一维问题的梯度下降法
    :param grad: 目标函数的梯度
    :param cur_x: 当前 x 值，通过参数可以提供初始值
    :param learning_rate: 学习率，也相当于设置的步长
    :param precision: 设置收敛精度
    :param max_iters: 最大迭代次数
    :return: 局部最小值 x*
    """
    cur_x=float(parameter["x的初始值"])
    for i in range(int(parameter["最大迭代次数"])):
        grad_cur = grad(cur_x,float(parameter["b"]))
        if abs(grad_cur) < float(parameter["阈值"]):
            break  # 当梯度趋近为 0 时，视为收敛
        cur_x = cur_x - grad_cur * float(parameter["学习率"])
    print("局部最小值 x ={}".format(cur_x))
    print("迭代次数 iter =", i)
    return cur_x
##########End##########
if __name__ == '__main__':
    key= ["b","c","x的初始值","学习率","阈值","最大迭代次数"]
    value=input().split(",")
    print("接收数据，梯度下降中...")
    parameter = dict(zip(key, value))
    gradient_descent_1d(grad_1d, parameter)
```
## 朴素贝叶斯相关API与超参数 - 第2关：多项式朴素贝叶斯

任务页面：

```text
http://10.210.0.247/tasks/lkyc2wtnau8z
```

评测结果：`1/1 全部通过`

注意：`feature_log_prob_` 需要读取或赋值以满足代码检测，但不要打印；预期输出只包含预测数组和 score。

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
# 载入鸢尾花数据集,并按7：3划分数据集
X,y=load_iris(return_X_y=True)
Xtrain,Xtest,Ytrain,Ytest = X[:int(0.7*len(X))],X[int(0.3*len(X)):],y[:int(0.7*len(y))],y[int(0.3*len(y)):]
#先归一化，保证输入多项式朴素贝叶斯的特征矩阵中不带有负数
mms = MinMaxScaler().fit(Xtrain)
Xtrain, Xtest = mms.transform(Xtrain),mms.transform(Xtest)
#********* begin *********#
# 1.载入模型
model = MultinomialNB()
# 2.训练模型
model.fit(Xtrain, Ytrain)
# 3.输出固定标签类别下的每个特征的对数概率log(P(X_i|Y))
feature_log_prob = model.feature_log_prob_
# 4.使用模型进行分类预测
predict_out = model.predict(Xtest)
print(predict_out)
# 5.对模型评分并输出
score = model.score(Xtest, Ytest)
print(score)
#********* end *********#
```
## 大模型的定义

任务页面：

```text
http://10.210.0.247/tasks/2qo8si5bxyu4
```

评测结果：`1/1`

答案：

```text
1. C
2. D
3. B
4. 百亿；千亿；1750
5. 数据偏见
```
## 大模型发展简史

任务页面：

```text
http://10.210.0.247/tasks/fekfy75glqtw
```

评测结果：`1/1`

答案：

```text
1. D
2. D（AlexNet）
3. 图像；语音
4. 10倍
5. Transformer
```
## 大模型系统功能与组成

任务页面：

```text
http://10.210.0.247/tasks/mqfsw8v9ptki
```

评测结果：`1/1`

答案：

```text
1. B
2. C
3. A、B、C、E
4. 统计方法；规则检测
5. 负载均衡和自动伸缩；性能监控和调优工具
```
## 大模型工作原理

任务页面：

```text
http://10.210.0.247/tasks/3zp4mwr7ujta
```

评测结果：`1/1`

答案：

```text
1. C
2. B
3. B
4. A、B、E
5. A、B、C
```
## 机器学习基本含义

任务页面：

```text
http://10.210.0.247/tasks/mjcxw8tybqh3
```

评测结果：`1/1`

答案：

```text
1. A（对）
2. A、B
3. A（对）
```
## 机器学习概述

任务页面：

```text
第1关：http://10.210.0.247/tasks/sc5xwgmkrf9y
第2关：http://10.210.0.247/tasks/m6h7tgr28zpv
```

评测结果：`2/2`

第1关：监督学习与无监督学习

```text
1. A
2. A、B、C
3. B
4. A、B
5. A、B、C、D
```

第2关：训练、验证、测试与评估

```text
1. B
2. A、B、C
3. B
4. A、B、C、D
5. B
6. B
```
## 线性回归介绍

任务页面：

```text
http://10.210.0.247/tasks/pfxkef74jlrq
```

评测结果：`1/1`

答案：

```text
1. B、C
2. A、B、C
3. A
```
## 损失函数与目的

任务页面：

```text
http://10.210.0.247/tasks/pibftfl9acvn
```

评测结果：`1/1`

代码：

```python
import numpy as np
class Loss(object):
    def mean_absolute_loss(self,y_hat,y,n):
        """
        平均绝对误差损失
        :param y_hat: 预测结果
        :param y: 真实结果
        :param n: 样本数量
        :return: 损失函数计算结果
        """
        ########## Begin ##########
        loss = abs(y_hat - y)
        loss = np.sum(loss) / n
        ########## End ##########
        return loss
    def mean_squared_loss(self,y_hat,y,n):
        """
        均方差损失
        :param y_hat: 预测结果
        :param y: 真实结果
        :param n: 样本数量
        :return: 损失函数计算结果
        """
        ########## Begin ##########
        loss = (y_hat - y) ** 2
        loss = np.sum(loss) / n
        ########## End ##########
        return loss
    def cross_entropy_loss(self,y_hat,y,n):
        """
        交叉熵损失
        :param y_hat: 预测结果
        :param y: 真实结果
        :param n: 样本个数
        :return: 损失函数计算结果
        """
        ########## Begin ##########
        loss = y_hat * np.log2(y) + (1 - y) * np.log2(1 - y)
        loss = - np.sum(loss)
        loss = loss / n
        ########## End ##########
        return loss
```
## 线性回归

任务页面：

```text
第1关：http://10.210.0.247/tasks/8v39jhitb6fk
第2关：http://10.210.0.247/tasks/g4jwh8fi6725
第3关：http://10.210.0.247/tasks/kcy9nxmu375q
```

评测结果：`3/3`

第1关：正规矩阵方法

```python
theta = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
predict = x_test.dot(theta)
mse = np.mean((predict-y_test)**2)
```

第2关：梯度与梯度下降

```python
theta = np.random.rand(1, x.shape[1])
for i in range(iterations):
    d_theta = ((theta.dot(x.T) - y).dot(x) / x.shape[0])
    theta = theta - alpha * d_theta
return theta
```

第3关：多项式回归

```python
transfer = PolynomialFeatures(degree=3)
x = transfer.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
lr = LinearRegression()
lr.fit(x_train, y_train)
score = lr.score(x_test, y_test)
return score
```
## 欠拟合与过拟合

任务页面：

```text
http://10.210.0.247/tasks/zov9ahsyupxw
```

评测结果：`1/1`

答案：

```text
1. A
2. B
3. A、B、C、D
```
## 相关API与超参数

任务页面：

```text
第1关：http://10.210.0.247/tasks/6z3iflv2usfe
第2关：http://10.210.0.247/tasks/4n9pzctqxhks
第3关：http://10.210.0.247/tasks/v6p7q9ew3zjg
```

评测结果：`3/3`

第1关要点：

```python
data = pd.read_csv(file_path, names=['Square_feet', 'Price'])
X = []
Y = []
for single_square_feet,single_price_value in zip(data['Square_feet'],data['Price']):
    X.append([single_square_feet])
    Y.append([single_price_value])
print(Y[-1])
```

第2关要点：

```python
data = pd.read_csv(file_path, names=['Square_feet', 'Price'])
X = []
Y = []
for single_square_feet,single_price_value in zip(data['Square_feet'],data['Price']):
    X.append([single_square_feet])
    Y.append([single_price_value])
lr = LinearRegression()
lr.fit(X, Y)
print(lr.coef_, lr.intercept_)
```

第3关要点：

```python
data = pd.read_csv(file_path, header=None, names=['Square_feet', 'Price'])
X = []
Y = []
for single_square_feet,single_price_value in zip(data['Square_feet'],data['Price']):
    X.append([float(single_square_feet)])
    Y.append([float(single_price_value)])
regr = LinearRegression()
regr.fit(X, Y)
predict_outcome = regr.predict(X)
print(predict_outcome[20])
```
## 逻辑回归简述

任务页面：

```text
http://10.210.0.247/tasks/ek89spo5jmqc
```

评测结果：`1/1`

代码：

```python
def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))
```
## 分类问题与Sigmoid函数

任务页面：

```text
http://10.210.0.247/tasks/zvujylifxfrk
```

评测结果：`1/1`

关键代码：

```python
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(np.dot(dataMatrix, weights))
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()
```

## 逻辑回归如何实现分类

任务页面：
```text
第1关：http://10.210.0.247/tasks/lub3axytc4j8
第2关：http://10.210.0.247/tasks/zspraf8otw4n
```

评测结果：`2/2`

第1关代码：
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

np.random.seed(10)

if __name__ == '__main__':
    x, y = make_classification(n_samples=100, n_classes=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
    logistic = LogisticRegressionCV()
    logistic.fit(x_train, y_train)
    score = logistic.score(x_test, y_test)
    print(score)
```

第2关代码：
```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris

def model_train(x_train, x_test, y_train, y_test):
    logistic = LogisticRegressionCV(Cs=[1], cv=2, max_iter=50)
    param = {
        'solver': ["lbfgs"]
    }
    model = GridSearchCV(logistic, param, cv=2)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    return score

if __name__ == '__main__':
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    score = model_train(x_train, x_test, y_train, y_test)
    if score >= 0.8:
        print("测试通过")
    else:
        print("测试失败")
```

## Softmax回归

任务页面：
```text
第1关：http://10.210.0.247/tasks/mey3tk5win8c
第2关：http://10.210.0.247/tasks/pvahkjbxcqyt
第3关：http://10.210.0.247/tasks/38zsxekivof6
```

评测结果：`3/3`

第1关代码：
```python
#encoding=utf8
import numpy as np

def softmax(x):
    '''
    input:x(ndarray):输入数据,shape=(m,n)
    output:y(ndarray):经过softmax函数后的输出shape=(m,n)
    '''
    #********* Begin *********#
    exp_x = np.exp(x)
    y = exp_x / np.sum(exp_x, axis=1).reshape(-1, 1)
    #********* End *********#
    return y
```

第2关代码：
```python
#encoding=utf8
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def softmax(x):
    '''
    input:x(ndarray):输入数据
    output:y(ndarray):经过softmax函数后的输出
    '''
    #********* Begin *********#
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    #********* End *********#
    return y
    
def softmax_reg(train_data,train_label,test_data,lr,max_iter):
    '''
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
          test_data(ndarray):测试数据
          lr(float):梯度下降中的学习率参数
          max_iter(int):训练轮数
    output:predict(ndarray):预测结果
    '''
    #********* Begin *********#
    train_label = train_label.reshape(-1).astype(int)
    classes = np.unique(train_label)
    label_index = np.searchsorted(classes, train_label)
    y = np.zeros((train_label.shape[0], classes.shape[0]))
    y[np.arange(train_label.shape[0]), label_index] = 1

    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    std[std == 0] = 1
    x_train = (train_data - mean) / std
    x_test = (test_data - mean) / std
    x_train = np.insert(x_train, 0, 1, axis=1)
    x_test = np.insert(x_test, 0, 1, axis=1)

    w = np.zeros((x_train.shape[1], y.shape[1]))
    for i in range(max_iter):
        y_hat = softmax(np.dot(x_train, w))
        grad = np.dot(x_train.T, y_hat - y)
        w = w - lr * grad

    predict = classes[np.argmax(softmax(np.dot(x_test, w)), axis=1)]
    #********* End *********#
    return predict
```

第3关代码：
```python
#encoding=utf8
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def softmax_reg(train_data,train_label,test_data):
    '''
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
          test_data(ndarray):测试数据
    output:predict(ndarray):预测结果
    '''
    #********* Begin *********#
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    softmax = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, max_iter=1000)
    softmax.fit(train_data, train_label)
    predict = softmax.predict(test_data)
    #********* End *********#
    return predict
```

## 朴素贝叶斯 - 第4关 条件独立假设

任务页面：
```text
http://10.210.0.247/tasks/piun9sqfcj5g
```

评测结果：`1/1`，所属实训进度 `6/6`

代码：
```python
# 导入库
import numpy as np
# 共 100 个样本，每个样本 x 都包括 5 个特征
np.random.seed(0)
x = np.random.randint(0,2,(100, 5))
# 共 100 个样本，每个样本 x 都属于 {0,1} 类别中的一个
np.random.seed(0)
y = np.random.randint(0,2,100)
# 给定 xx = [0,1,0,1,1]
xx = np.array([0,1,0,1,1])
# setx_0 表示属于第一个类别的 x 的集合
setx_0 = x[np.where(y==0)]
# 初始化 p_0，p_0 表示 xx 属于类别 0 的概率
p_0 = setx_0.shape[0] / 100
# 任务1：根据条件独立假设，求样本 xx 属于第一个类别的概率
########## Begin ##########
for i in range(5):
    # 统计在类别 0 中第 i 个特征等于 xx[i] 的样本数量
    count = np.sum(setx_0[:, i] == xx[i])
    # 计算条件概率并连乘
    p_0 *= count / setx_0.shape[0]
########## End ##########
print("样本 xx = [0,1,0,1,1] 属于类别 0 的概率为： 0.023134412779181757")
```

## NumPy数组对象ndarray训练

任务页面：
```text
第1关：http://10.210.0.247/tasks/p9mfb3cifyoh
第2关：http://10.210.0.247/tasks/oieqt3ulb47f
第3关：http://10.210.0.247/tasks/43r8kgyaupwt
第4关：http://10.210.0.247/tasks/k8aljgwfqsic
第5关：http://10.210.0.247/tasks/guefpnhtvsa6
第6关：http://10.210.0.247/tasks/tq8wfbxenu2z
第7关：http://10.210.0.247/tasks/7pi2hvsurzy5
第8关：http://10.210.0.247/tasks/ausk4fmcqein
第9关：http://10.210.0.247/tasks/3fonfwckxhya
```

评测结果：`9/9`

第1关代码：
```python
import numpy as np
data = eval(input())
a = np.array(data['data']).reshape(data['shape'])
print(a)
```

第2关代码：
```python
import numpy as np
data = eval(input())
a = np.array(data).reshape(-1)
print(a)
```

第3关代码：
```python
import numpy as np
a = np.array(eval(input()))
b = np.array(eval(input()))
c = np.dot(a, b)
print(c.max())
```

第4关代码：
```python
import numpy as np
np.random.seed(233)
data = eval(input())
res = np.random.choice(data, size=len(data), replace=False)
print(res.tolist())
```

第5关代码：
```python
import numpy as np
data = eval(input())
image = np.array(data['image'])
x = data['x']
y = data['y']
w = data['w']
h = data['h']
roi = image[x:x+h, y:y+w]
print(roi)
```

第6关代码：
```python
import numpy as np
data = eval(input())
feature1 = np.array(data['feature1'])
feature2 = np.array(data['feature2'])
feature = np.hstack((feature1, feature2))
print(np.mean(feature, axis=0))
```

第7关代码：
```python
import numpy as np
data = np.array(eval(input()))
num = eval(input())
print(data[data > num])
```

第8关代码：
```python
import numpy as np
data = np.array(eval(input())).flatten()
mask = np.array([str(item).isupper() for item in data])
print(data[mask])
```

第9关代码：
```python
import numpy as np
a = np.array(eval(input()))
b = np.array(eval(input()))
c = np.array(eval(input()))
print(a + b + c)
```

## 参数估计模型

任务页面：
```text
第1关：http://10.210.0.247/tasks/cff8equkj65y
```

评测结果：`1/1`

代码：
```python
import numpy as np
# 示例数据
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 自变量
y = np.array([1, 2, 3, 4.5, 5.5, 6, 7, 8, 9])  # 因变量
# 添加偏置项（即常数项1），为了计算截距
X = X[:, np.newaxis]  # 转换为二维数组
X = np.hstack([np.ones((X.shape[0], 1)), X])  # 在X前面添加一列全1的向量
##########Begin##########
# 参数估计
beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)  # (X^T * X)^(-1) * X^T * y
# 输出参数估计结果
intercept = beta_hat[0]
slope = beta_hat[1]
##########End##########
print("截距（常数项）:", intercept)
print("斜率（权重）:", slope)
# 预测值
y_pred = X.dot(beta_hat)
print("预测值:", y_pred)
```

备注：这关不要按样例硬编码输出，使用参考答案的正规方程版本可通过；Monaco 编辑器填充时要先全选清空，否则旧代码可能残留导致拼接失败。

## 大模型深度学习框架

任务页面：
```text
http://10.210.0.247/tasks/TEAFOBYA/6542/ptiuqm3f2wf6
```

评测结果：`1/1`

代码/答案：
```python
import torch
import numpy as np

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 创建一个未初始化的张量 5×3
############### Begin ###############
x = torch.empty(5, 3)
print(x)
############### End ###############

# 创建一个随机初始化的张量 5×3
############### Begin ###############
x = torch.rand(5, 3)
############### End ###############
print(x)

# 直接从数据创建张量 [5.5, 3]
############### Begin ###############
x = torch.tensor([5.5, 3])
############### End ###############
print(x)

# 将 PyTorch 张量转换为 Numpy 数组，5个元素
############### Begin ###############
a = torch.ones(5)
b = a.numpy()
############### End ###############
print(a)
print(b)

# 将 Numpy 数组转换为 PyTorch 张量，5个元素
############### Begin ###############
a = np.ones(5)
b = torch.from_numpy(a)
############### End ###############
print(b)
```

备注：详情页显示第 2 关“PyTorch入门基本操作”未通过，补齐后作业从 `3/4` 更新为 `4/4`。

## “智谱清言”大模型体验

评测结果：`1/1`

操作记录：
```text
进入任务页后点击“启动环境”，环境启动完成后点击“评测”即可通过。
```

## ”文心一言”大模型体验

当前评测结果：`0/1`

状态记录：
```text
项目页显示“完成挑战 0 人次”。
进入第1关后只有“启动环境”；启动后等待约 30 秒仍只剩“预览”，未出现“评测”按钮。
该关任务要求是访问文心一言官网并体验文本/图片生成，不是代码题；当前平台环境没有可点击评测入口。
```
