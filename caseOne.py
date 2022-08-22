import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np


#创建测试数据
x_data = np.random.rand(300).astype(np.float32)
y_data = x_data * 0.2 + 0.3
###-------创建结构开始----------###

#搭建模型
Weights = tf.Variable(tf.random.uniform([1],-1,1.0))
biases = tf.Variable(tf.zeros([1]))
print(Weights)

y = x_data * Weights + biases

#计算误差
loss = tf.reduce_mean(tf.square(y-y_data))


#反向传递，优化权重偏置
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


###————————创建结构结束—————————###

# 初始化结构
init = tf.global_variables_initializer()
# 获取Session
sess = tf.Session()
#用Session进行初始化
sess.run(init)

# 开始训练
for step in range(300):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))


