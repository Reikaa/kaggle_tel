__author__ = 'Thushan Ganegedara'

import csv
import collections
import numpy as np

probs_1 = collections.defaultdict()
probs_2 = collections.defaultdict()

logloss_1 = 0.467649
acc_1 = 1- logloss_1
ids_with_over_90_1 = []

logloss_2 = 0.68407
acc_2 = 1 - logloss_2
ids_with_over_90_2 = []
# XGBOOST
with open('xgboost_output.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for i,row in enumerate(reader):
        probs_1[int(row[0])] = [float(row[1]),float(row[2]),float(row[3])]
        if np.max(probs_1[int(row[0])])>0.9:
            ids_with_over_90_1.append(int(row[0]))

# SDAE
with open('deepnet_out_probs.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for i,row in enumerate(reader):
        probs_2[int(row[0])] = [float(row[1]),float(row[2]),float(row[3])]
        if np.max(probs_2[int(row[0])])>0.9:
            ids_with_over_90_2.append(int(row[0]))

ens_probs = collections.defaultdict()
for k,v in probs_1.items():
    if k in ids_with_over_90_1 or k in ids_with_over_90_2:
        vec1 = [x * (acc_1/(acc_1+acc_2)) for x in probs_1[k] ]
        vec2 = [y * (acc_2/(acc_1+acc_2)) for y in probs_2[k]]

        vec3 = [x+y for x,y in zip(vec1,vec2)]
    else:
        vec3 = v


    ens_probs[k] = vec3

correct_order_test_ids = []
with open('test.csv', 'r',newline='') as f:
        reader = csv.reader(f)

        for i,row in enumerate(reader):
            if i==0:
                continue
            correct_order_test_ids.append(int(row[0]))

with open('ensemble_probs.csv', 'w',newline='') as f:
        writer = csv.writer(f)

        for k in correct_order_test_ids:
            row = [k]
            row.extend(ens_probs[k])

            writer.writerow(row)

