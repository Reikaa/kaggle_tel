__author__ = 'Thushan Ganegedara'

import csv
import collections
import numpy as np

train_data = collections.defaultdict()
all_locs = []
with open('train.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for i,row in enumerate(reader):
        if i==0:
            continue
        data_row = []
        id,loc,sev = None,None,None

        for j,col in enumerate(row):
            if j==0:
                id = col
            elif j==1:
                loc = int(col[9:])
                all_locs.append(int(col[9:]))
            elif j==2:
                sev = int(col)
        train_data[id]=[loc,sev]

max_loc = np.max(all_locs)
train_count = len(train_data)

test_data = collections.defaultdict()
with open('test.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for i,row in enumerate(reader):
        if i==0:
            continue
        data_row = []
        id,loc = None,None

        for j,col in enumerate(row):
            if j==0:
                id = col
            elif j==1:
                loc = int(col[9:])
                all_locs.append(int(col[9:]))
        test_data[id]=[loc]

feature_data = collections.defaultdict()
all_features = []
max_per_feature = collections.defaultdict()
feature_count = collections.defaultdict()

with open('log_feature.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for i,row in enumerate(reader):
        if i==0:
            continue
        data_row = []
        id,feature,vol = None,None,None


        for j,col in enumerate(row):
            if j==0:
                id = col
            elif j==1:
                feature = int(col[8:])
                all_features.append(int(col[8:]))
            elif j==2:
                vol = int(col)

        if feature in feature_count:
            feature_count[feature] += 1
        else:
            feature_count[feature] = 1

        if id in feature_data:
            feature_data[id].append([feature,vol])
        else:
            feature_data[id]=[[feature,vol]]

        if feature in max_per_feature:
            if vol > max_per_feature[feature]:
                max_per_feature[feature] = vol
        else:
            max_per_feature[feature] = vol


important_features=[] # based on the values count for each feature
for k,v in feature_count.items():
    if v > 50:
        important_features.append(k)
max_feature = len(important_features)

severity_data = collections.defaultdict()
all_severity = []

with open('severity_type.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for i,row in enumerate(reader):
        if i==0:
            continue
        data_row = []
        id,sev = None,None

        for j,col in enumerate(row):
            if j ==0:
                id = col
            elif j==1:
                sev = int(col[13:])
                all_severity.append(sev)
        severity_data[id]=[sev]

max_severity = np.max(all_severity)

event_data = collections.defaultdict()
all_event = []

with open('event_type.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for i,row in enumerate(reader):
        if i==0:
            continue
        data_row = []
        id,event = None,None

        for j,col in enumerate(row):
            if j ==0:
                id = col
            elif j == 1:
                event = int(col[11:])
                all_event.append(event)
        event_data[id] = [event]

max_event = np.max(all_event)

resource_data = collections.defaultdict()
all_resource =[]
with open('resource_type.csv', 'r',newline='') as f:
    reader = csv.reader(f)

    for i,row in enumerate(reader):
        if i==0:
            continue
        data_row = []
        id,res = None,None

        for j,col in enumerate(row):
            if j ==0:
                id = col
            elif j == 1:
                res = int(col[14:])
                all_resource.append(res)
        resource_data[id] = [res]

max_res = np.max(all_resource)

'''
#concat all data
all_data = collections.defaultdict()
all_out = collections.defaultdict()
all_feature_val = collections.defaultdict()
for k,v in train_data.items():
    row = [v[0]] # we need to add the output to a different dic
    all_out[k] = v[1]
    if k in feature_data:
        v.extend(feature_data[k][0])
        #all_feature_val[k] = feature_data[k][1]
    if k in severity_data:
        v.extend(severity_data[k])
    if k in event_data:
        v.extend(event_data[k])
    if k in resource_data:
        v.extend(resource_data[k])

    all_data[k] = v
    assert len(v)==7'''

neuron_count = (1) + (max_feature+1) + (max_severity+1) + (max_event+1) + (max_res+1)
print('neuron count: ',neuron_count)

def turn_to_vec(val_idx,max_val,val=1):
    row = [0 for _ in range(max_val+1)]
    row[val_idx] = val
    return row

valid_set = []
with open('features_train_vectorized.csv', 'w',newline='') as f:
        writer = csv.writer(f)
        for k,v in train_data.items():
            write_row=[]
            out = v[1]
            print('vectorizing location')
            #loc_vec = turn_to_vec(v[0],max_loc)
            loc_vec = [v[0]]

            print('vectorizing features')
            f_list = feature_data[k]
            feature_vec = [0 for _ in range(max_feature+1)]
            # list[0] is feature id and list[1] is corresponding value
            for list in f_list:
                if list[0] in important_features:
                    feature_vec[important_features.index(list[0])] = list[1]*1.0/max_per_feature[list[0]]
                    assert feature_vec[important_features.index(list[0])]<=1


            print('vectorizing severity')
            sev_val = severity_data[k]
            sev_vec = turn_to_vec(sev_val[0],max_severity)

            print('vectorizing event')
            event_val = event_data[k]
            print (event_val,',',max_event)
            event_vec = turn_to_vec(event_val[0],max_event)

            print('vectorizing resources')
            res_val = resource_data[k]
            res_vec = turn_to_vec(res_val[0],max_res)

            write_row = [k]
            write_row.extend(loc_vec)
            write_row.extend(feature_vec)
            write_row.extend(sev_vec)
            write_row.extend(event_vec)
            write_row.extend(res_vec)
            write_row.extend([out])
            print('len write row: ',len(write_row))
            assert len(write_row)==neuron_count+2

            writer.writerow(write_row)

with open('features_train.csv', 'w',newline='') as f:
        header = ['id','location']
        for feat in important_features:
            header.append('feature_'+str(feat))
        header.extend(['severity','event','resource','output'])

        writer = csv.writer(f)
        writer.writerow(header)
        for k,v in train_data.items():
            write_row = [k]
            out = v[1]
            write_row.append(v[0]) #location

            f_list = feature_data[k] #list of [feature_id, value] for a particular entry ID
            feature_vec = [0 for _ in range(max_feature)]
            # list[0] is feature id and list[1] is corresponding value
            for list in f_list:
                if list[0] in important_features:
                    feature_vec[important_features.index(list[0])] = list[1]*1.0/max_per_feature[list[0]]

            write_row.extend(feature_vec)

            write_row.append(severity_data[k][0]*1.0/max_severity)
            write_row.append(event_data[k][0]*1.0/max_event)
            write_row.append(resource_data[k][0]*1.0/max_res)
            write_row.append(out)
            writer.writerow(write_row)

with open('features_test_vectorized.csv', 'w',newline='') as f:
        writer = csv.writer(f)
        for k,v in test_data.items():
            write_row=[]

            print('vectorizing location')
            #loc_vec = turn_to_vec(v[0],max_loc)\
            loc_vec = [v[0]]

            print('vectorizing features')
            f_list = feature_data[k]
            feature_vec = [0 for _ in range(max_feature+1)]
            # list[0] is feature id and list[1] is corresponding value
            for list in f_list:
                if list[0] in important_features:
                    feature_vec[important_features.index(list[0])] = list[1]*1.0/max_per_feature[list[0]]
                    assert feature_vec[important_features.index(list[0])]<=1

            print('vectorizing severity')
            sev_val = severity_data[k]
            sev_vec = turn_to_vec(sev_val[0],max_severity)

            print('vectorizing event')
            event_val = event_data[k]
            print (event_val,',',max_event)
            event_vec = turn_to_vec(event_val[0],max_event)

            print('vectorizing resources')
            res_val = resource_data[k]
            res_vec = turn_to_vec(res_val[0],max_res)

            write_row = [k]
            write_row.extend(loc_vec)
            write_row.extend(feature_vec)
            write_row.extend(sev_vec)
            write_row.extend(event_vec)
            write_row.extend(res_vec)
            print(len(write_row))
            assert len(write_row)==neuron_count+1
            writer.writerow(write_row)

with open('features_test.csv', 'w',newline='') as f:

        header = ['id','location']
        for feat in important_features:
            header.append('feature_'+str(feat))
        header.extend(['severity','event','resource'])

        writer = csv.writer(f)
        writer.writerow(header)
        for k,v in test_data.items():
            write_row=[k]

            write_row.append(v[0]) #location

            f_list = feature_data[k] #list of [feature_id, value] for a particular entry ID
            feature_vec = [0 for _ in range(max_feature)]
            # list[0] is feature id and list[1] is corresponding value
            for list in f_list:
                if list[0] in important_features:
                    feature_vec[important_features.index(list[0])] = list[1]*1.0/max_per_feature[list[0]]

            write_row.extend(feature_vec)

            write_row.append(severity_data[k][0]*1.0/max_severity)
            write_row.append(event_data[k][0]*1.0/max_event)
            write_row.append(resource_data[k][0]*1.0/max_res)

            #assert len(write_row)==neuron_count
            writer.writerow(write_row)