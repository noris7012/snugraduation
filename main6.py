#! /home/noris/tensorflow/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

xy_raw = np.loadtxt('data_mul5.csv', delimiter=',', dtype=np.float32)

xy = xy_raw / xy_raw.max(axis=0)

train_size = int(len(xy) * 0.8)

# 1. 전체결제 * 세부 변수
x_idx = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32]
y_idx = [33]

# 2. 전체결제 * 전체 변수
# x_idx = [10, 26, 27, 28, 29, 30]
# y_idx = [33]

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
# x_idx = [10]
# y_idx = [3]

# 8. 전체결제 * 업적 수
# x_idx = [26]
# y_idx = [2]

# 9. 전체결제 * 아이템 수
# x_idx = [27]
# y_idx = [3]

# 10. 전체결제 * 친구 수
# x_idx = [28]
# y_idx = [3]

# 11. 전체결제 * 길드원 수
# x_idx = [29]
# y_idx = [3]

# 12. 전체결제 * 접속 빈도수
# x_idx = [30]
# y_idx = [3]

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
y_data = xy_raw[:train_size, y_idx]

nb_classes = 7 # 0 ~ 5

X = tf.placeholder(tf.float32, shape=[None, len(x_idx)])
Y = tf.placeholder(tf.int32, shape=[None, len(y_idx)])

Y_one_hot = tf.one_hot(Y, nb_classes)
print("one_hot", Y_one_hot)

Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)

W = tf.Variable(tf.random_normal([len(x_idx), nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# hypothesis = tf.matmul(X, W) + b
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=5.0).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

last_cost = 0

for step in range(20001):
    w_val, b_val, _ = sess.run([W, b, optimizer], feed_dict={X: x_data, Y: y_data})
    if step % 1000 == 0:
        loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
        print("W:",w_val,", b:",b_val)

        if loss == last_cost:
            break;

        last_cost = loss;

#x_test = xy[train_size:, [10, 28, 29, 30, 31, 32]]
x_test = xy[train_size:, x_idx]
y_test = xy_raw[train_size:, y_idx]

# Let's see if we can predict
pred = sess.run(prediction, feed_dict={X: x_test})
# y_data: (N,1) = flatten => (N, ) matches pred.shape

cnt_0 = 0
cnt_1 = 0
cnt_2 = 0
cnt_3 = 0
cnt_4 = 0
cnt_5 = 0
cnt_6 = 0
s_0 = 0
s_1 = 0
s_2 = 0
s_3 = 0
s_4 = 0
s_5 = 0
s_6 = 0
for p, y in zip(pred, y_test.flatten()):
    if y == 0:
        cnt_0 = cnt_0 + 1
        if p == (int(y)):
            s_0 = s_0 + 1

    if y == 1:
        cnt_1 = cnt_1 + 1
        if p == (int(y)):
            s_1 = s_1 + 1

    if y == 2:
        cnt_2 = cnt_2 + 1
        if p == (int(y)):
            s_2 = s_2 + 1

    if y == 3:
        cnt_3 = cnt_3 + 1
        if p == (int(y)):
            s_3 = s_3 + 1

    if y == 4:
        cnt_4 = cnt_4 + 1
        if p == (int(y)):
            s_4 = s_4 + 1

    if y == 5:
        cnt_5 = cnt_5 + 1
        if p == (int(y)):
            s_5 = s_5 + 1

    if y == 6:
        cnt_6 = cnt_6 + 1
        if p == (int(y)):
            s_6 = s_6 + 1

print ("len", len(y_test), "acc:", float(s_0+s_1+s_2+s_3+s_4+s_5+s_6)/(cnt_0+cnt_1+cnt_2+cnt_3+cnt_4+cnt_5+cnt_6)) 

print ("c0: {:5}\ts0: {:5}\tf0: {:5}\tacc: {:.2%}".format(cnt_0, s_0, cnt_0-s_0, float(s_0)/cnt_0))
print ("c1: {:5}\ts1: {:5}\tf1: {:5}\tacc: {:.2%}".format(cnt_1, s_1, cnt_1-s_1, float(s_1)/cnt_1))
print ("c2: {:5}\ts2: {:5}\tf2: {:5}\tacc: {:.2%}".format(cnt_2, s_2, cnt_2-s_2, float(s_2)/cnt_2))
print ("c3: {:5}\ts3: {:5}\tf3: {:5}\tacc: {:.2%}".format(cnt_3, s_3, cnt_3-s_3, float(s_3)/cnt_3))
print ("c4: {:5}\ts4: {:5}\tf4: {:5}\tacc: {:.2%}".format(cnt_4, s_4, cnt_4-s_4, float(s_4)/cnt_4))
print ("c5: {:5}\ts5: {:5}\tf5: {:5}\tacc: {:.2%}".format(cnt_5, s_5, cnt_5-s_5, float(s_5)/cnt_5))
print ("c6: {:5}\ts6: {:5}\tf6: {:5}\tacc: {:.2%}".format(cnt_6, s_6, cnt_6-s_6, float(s_6)/cnt_6))
