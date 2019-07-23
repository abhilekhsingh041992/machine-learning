# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/data_description.txt"))


# Any results you write to the current directory are saved as output.

"""
MSSubClass
MSZoning
LotFrontage
LotArea
Street
Alley
"""

TRAIN_CSV_PATH = "../input/train.csv"
TEST_CSV_PATH = "../input/test.csv"

train_data = pd.read_csv(TRAIN_CSV_PATH)
test_data = pd.read_csv(TEST_CSV_PATH)

headers = train_data.columns.values
#print(headers)
    
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
    
w = tf.Variable([0.2, 0.3], name="weights") 
model_y = tf.multiply([x1, x2], w)

error = tf.sqrt(tf.square(y - model_y))
# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

train_data = train_data
with tf.Session() as sess:
    init = tf.global_variables_initializer()                                   
    sess.run(init) 
       
    for i in range(10):
        for row in train_data.iterrows():
            house = row[1]

            expect_output = house["SalePrice"]

            #print(house["MSSubClass"], house["LotArea"])
            
            _, err = sess.run([train_op, error], feed_dict={x1: house["MSSubClass"], x2: house["LotArea"],
                                                            y: expect_output})

            #print(input_val, expect_output, err)
        
        #print(sess.run(w))  
        
    ids = []
    SalePrices = []
    for row in test_data.iterrows():
        house = row[1]

        SalePrice = (house["MSSubClass"] * sess.run(w[0])) + (house["LotArea"] * sess.run(w[1]))
        
        ids.append(house["Id"])
        SalePrices.append(SalePrice)
        
        #print(house["Id"], SalePrice)
        
    data_to_submit = pd.DataFrame({
        'Id': ids,
        'SalePrice': SalePrices
    })
    data_to_submit.to_csv('out.csv',index=False)
    
        






