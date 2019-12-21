import numpy as np
data = np.array([[10,3,9,1],[9,1,7,1],[4,0,5.5,0],[6,1,8,1]])
data[:,3]
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(data[:,0],data[:,1])
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
data_std = st.fit_transform(data[:,:3])
lr = linear_model.LogisticRegression()
lr
lr.fit(data_std,data[:,3])
lr.coef_
lr.intercept_
plt.scatter(data_std[:,0],data_std[:,1])
plt.plot(data_std[:,0],0.4798098*data_std[:,0]+0.56706637)

'''
对于线性回归和逻辑回归，其目标函数为：

g(x) = w1x1 + w2x2 + w3x3 + w4x4 + w0

如果有激活函数sigmoid，增加非线性变化? 则为分类? 即逻辑回归

如果没有激活函数，则为回归

对于这样的线性函数，都会有coef_和intercept_函数

如下：

lr = LogisticRegression()

lr.coef_

lr.intercept_

coef_和intercept_都是模型参数，即为w

coef_为w1到w4

intercept_为w0
'''
