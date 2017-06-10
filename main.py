#! /home/noris/tensorflow/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

xy_raw = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)

xy = xy_raw / xy_raw.max(axis=0)

train_size = int(len(xy) * 0.8)

# 1. 전체결제 * 세부 변수
# x_idx = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32]
# y_idx = [1]

# 2. 전체결제 * 전체 변수
# x_idx = [10, 26, 27, 28, 29, 30]
y_idx = [3]

# 3. 선물 결제 * 세부 변수
# x_idx = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32]
# y_idx = [5]

# 4. 선물 결제 * 전체 변수
# x_idx = [10, 26, 27, 28, 29, 30]
# y_idx = [5]

# 5. 자기 결제 * 세부 변수
# x_idx = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32]
# y_idx = [8]

# 6. 자기 결제 * 전체 변수
# x_idx = [10, 26, 27, 28, 29, 30]
# y_idx = [8]

# 7. 전체결제 * 플레이 타임
#x_idx = [10]
# y_idx = [1]

# 8. 전체결제 * 업적 수
# x_idx = [26]
# y_idx = [1]

# 9. 전체결제 * 아이템 수
# x_idx = [27]
# y_idx = [1]

# 10. 전체결제 * 친구 수
# x_idx = [28]
# y_idx = [2]

# 11. 전체결제 * 길드원 수
# x_idx = [29]
# y_idx = [2]

# 12. 전체결제 * 접속 빈도수
x_idx = [30]
# y_idx = [2]

# 13. 전체결제 * 세부 플레이 타임
# x_idx = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
# y_idx = [2]

# 14. 전체결제 * 세부 플레이 타임
# x_idx = [31, 32]
# y_idx = [2]

# x_idx = [10, 26, 28, 29, 30]
# y_idx = [2]

# 플레이, 엠블럼, 아이템, 친구, 클랜, 접속 빈도
# x_data = xy[:train_size, [10, 28, 29, 30, 31, 32]]
x_data = xy[:train_size, x_idx]
y_data = xy[:train_size, y_idx]

X = tf.placeholder(tf.float32, shape=[None, len(x_idx)])
Y = tf.placeholder(tf.float32, shape=[None, len(y_idx)])

W = tf.Variable(tf.random_normal([len(x_idx), len(y_idx)]), name='weight')
b = tf.Variable(tf.random_normal([len(y_idx)]), name='bias')
c = tf.Variable(tf.random_normal([len(y_idx)]), name='bias2')

#hypothesis = tf.matmul(X, W) + b
hypothesis = tf.exp(tf.matmul(X, W) + b) + c

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

last_cost = 0

for step in range(100001):
    cost_val, hy_val, w_val, b_val, _ = sess.run([cost, hypothesis, W, b, train], feed_dict={X: x_data, Y: y_data})
    if step % 10000 == 0:
        print(step, "Cost: ", cost_val, ", b: ", b_val)
        print("W: ", w_val)
        print("Prediction: ", hy_val[:3] * 998910)

        if last_cost == cost_val:
            break

        last_cost = cost_val

print ("train done")

#x_test = xy[train_size:, [10, 28, 29, 30, 31, 32]]
x_test = xy[train_size:, x_idx]
y_test = xy[train_size:, y_idx]
hy_test = sess.run(hypothesis, feed_dict={X: x_test})

rate = (hy_test - y_test) / y_test
rate2 = (hy_test - y_test) * 8464940

ccc = np.count_nonzero((rate > -0.1) & (rate < 0.1))
print ("np.count_nonzero((rate > -0.1) & (rate < 0.1))")
print (ccc)

print ("np.average(rate)")
print (np.average(rate))

print ("len(rate)")
print (len(rate))

print ("accuracy")
print (float(ccc) / float(len(rate)) * 100.0)
