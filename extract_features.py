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
min_loc = np.min(all_locs)

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
    if v > 10:
        important_features.append(k)
max_feature = np.max(all_features)
min_feature = 0
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
min_severity = np.min(all_severity)

event_data = collections.defaultdict()
event_count = collections.defaultdict()
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

        if event in event_count:
            event_count[event] += 1
        else:
            event_count[event] = 1

        if id in event_data:
            event_data[id].append(event)
        else:
            event_data[id] = [event]

important_events = []
for k,v in event_count.items():
    if v>10:
        important_events.append(k)

max_event = np.max(all_event)
min_event = np.min(all_event)

max_events_per_id = 0
for k,v in event_data.items():
    if len(v)>max_events_per_id:
        max_events_per_id = len(v)

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
                all_resource.append(res-1)
        if id in resource_data:
            resource_data[id].append(res-1)
        else:
            resource_data[id] = [res-1]

max_res = np.max(all_resource)
min_res = np.min(all_resource)

max_res_per_id = 0
for k,v in resource_data.items():
    if len(v)>max_res_per_id:
        max_res_per_id = len(v)

# features start with 1
# severity start with 1
# event starts with 1
# resource starts with 1
# 2 for id and location
neuron_count = 2 + (max_feature) + (max_severity) + (max_event) + (max_res)
print('neuron count: ',neuron_count)
print('Loc: ',min_loc,', ',max_loc)
print('Features: ',0,', ',max_feature,' (',len(important_features),')')
print('Event: ',min_event,',',max_event,', (',len(important_events),')')
print('Severity: ',min_severity,', ',max_severity)
print('Res: ',min_res,', ',max_res)

def turn_to_vec(indices,max_val,val=1):
    row = [0 for _ in range(max_val+1)]
    for i in indices:
        row[i] = val
    return row

valid_set = []


def write_severity_in_order(file_name,train_data,test_data):

    sev_file_name = file_name + '_severity.csv'
    severity_data = []
    loc_order_by_sev = {}
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
            severity_data.append((id,sev))

    prev_diff_loc = -1
    order = 0
    for id,sev in severity_data:
        if id in train_data:
            curr_loc = train_data[id][0]
        else:
            curr_loc = test_data[id][0]

        if curr_loc != prev_diff_loc:
            order = 0

        loc_order_by_sev[id]=order
        order += 1
        prev_diff_loc = curr_loc

    with open(sev_file_name, 'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','sev','location','fault','order'])

        for id,sev in severity_data:

            if id in train_data:
                row = [str(id),sev,str(train_data[id][0]),str(train_data[id][1]),str(loc_order_by_sev[id])]
            else:
                row = [str(id),sev,str(test_data[id][0]),-1,str(loc_order_by_sev[id])]

            writer.writerow(row)


def write_file(file_name,train_data,feature_data,severity_data,event_data,resource_data,include,isTrain=True,noise=False):
    if isTrain:
        full_file_name = file_name+'_train.csv'
    else:
        full_file_name = file_name + '_test.csv'

    train_data_mat = [[],[],[]]

    loc_vec,feature_vec,event_vec,sev_vec,res_vec = None,None,None,None,None

    doOnce = False
    with open(full_file_name, 'w',newline='') as f:
        writer = csv.writer(f)
        for k,v in train_data.items():

            if 'id' in include:
                write_row = [k]
            else:
                write_row = []

            loc_thresh = 100 # 10 will give 110 element vec, 100 will give 11 element vec
            if 'loc' in include:
                if 's' in include['loc'] :
                    if 'n' in include['loc']:
                        loc_vec = [v[0]*1.0/max_loc]
                    else:
                        loc_vec = [v[0]]
                elif 'v' in include['loc']:
                    from math import floor,ceil
                    val_for_v0 = float((v[0]*1.0 % loc_thresh)/loc_thresh)
                    idx_for_v0 = []

                    idx_for_v0 = [floor(v[0]/loc_thresh)]
                    if loc_thresh!=1:
                        loc_vec = turn_to_vec(idx_for_v0,floor(max_loc*1./loc_thresh),val_for_v0)
                    else:
                        loc_vec = turn_to_vec([v[0]],max_loc,1)


            if 'feat' in include:
                f_list = feature_data[k]

                # f_list is like [[feat_x,val],[feat_y,val],...]
                # list[0] is feature id and list[1] is corresponding value

                if 'bins' in include['feat']:
                    bin_size = 10
                    feature_vec = [0 for _ in range(ceil((max_feature+1)/bin_size))]

                    # f_list is feature list for k-th id
                    for list in f_list:
                        idx = int(floor(list[0]/bin_size))
                        if 'n' in include['feat']:
                            raise NotImplementedError
                        else:
                            feature_vec[idx] += list[1]


                else:
                    feature_vec = [0 for _ in range(max_feature+1)]
                    for list in f_list:
                        if list[0] in important_features:
                            if 'n' in include['feat']:
                                feature_vec[important_features.index(list[0])] = list[1]*1.0/max_per_feature[list[0]]
                                assert feature_vec[important_features.index(list[0])]<=1
                            else:
                                if 'mul_sev' in include['feat']:
                                    feature_vec[important_features.index(list[0])] = list[1] * severity_data[k][0]
                                else:
                                    feature_vec[important_features.index(list[0])] = list[1]

            if 'sev' in include:
                if 's' in include['sev']:
                    if 'n' in include['sev']:
                        sev_vec = [severity_data[k]*1.0/max_severity]
                    else:
                        sev_vec = [severity_data[k][0]]
                elif 'v' in include['sev']:
                    sev_vec = turn_to_vec(severity_data[k],max_severity)

            if 'eve' in include:
                if 's' in include['eve']:
                    event_vec = [-1 for _ in range(max_events_per_id)]
                    for tmp_i,e in enumerate(event_data[k]):
                        event_vec[tmp_i] = e
                    if 'mul' in include['eve']:
                        e_mul = 1
                        for e in event_data[k]:
                            e_mul *= e
                        event_vec.append(e_mul)
                    if 'mu' in include['eve']:
                        e_mu = np.mean(event_data[k])
                        event_vec.append(e_mu)

                elif 'v' in include['eve']:
                    if 'bins' in include['eve']:
                        bin_size = 5
                        e_bin_ids = []
                        vec_size = int(ceil(max_event/bin_size))
                        for e_id in event_data[k]:
                            tmp_id = int(floor(e_id/bin_size))
                            if tmp_id not in e_bin_ids:
                                e_bin_ids.append(tmp_id)
                        event_vec = turn_to_vec(e_bin_ids,vec_size)
                    else:
                        event_vec = turn_to_vec(event_data[k],max_event)

            if 'res' in include:
                if 's' in include['res']:
                    res_vec =[-1 for _ in range(max_res_per_id)]
                    for tmp_i,r in enumerate(resource_data[k]):
                        res_vec[tmp_i] = r
                    if 'mul' in include['res']:
                        r_mul = 1
                        for r in resource_data[k]:
                            r_mul *= r
                        res_vec.append(r_mul)
                    if 'mu' in include['res']:
                        r_mu = np.mean(resource_data[k])
                        res_vec.append(r_mu)
                elif 'v' in include['res']:
                    if 'mul_sev' in include['res']:
                        res_vec = turn_to_vec(resource_data[k],max_res,severity_data[k][0])
                    else:
                        res_vec = turn_to_vec(resource_data[k],max_res)


            if not doOnce:
                header = ['id']
                if loc_vec is not None:
                    header.extend(['loc_'+str(i) for i in range(len(loc_vec))])
                if feature_vec is not None:
                    header.extend(['feat_'+str(i) for i in range(len(feature_vec))])
                if sev_vec is not None:
                    header.extend(['sev_'+str(i) for i in range(len(sev_vec))])
                if event_vec is not None:
                    header.extend(['eve_'+str(i) for i in range(len(event_vec))])
                if res_vec is not None:
                    header.extend(['res_'+str(i) for i in range(len(res_vec))])
                if isTrain:
                    header.append('out')
                writer.writerow(header)
                doOnce = True

            if loc_vec is not None:
                write_row.extend(loc_vec)
            if feature_vec is not None:
                write_row.extend(feature_vec)
            if sev_vec is not None:
                write_row.extend(sev_vec)
            if event_vec is not None:
                write_row.extend(event_vec)
            if res_vec is not None:
                write_row.extend(res_vec)

            if noise:
                noise_vec = np.random.binomial(1,0.25,(len(write_row)))*np.random.random((len(write_row)))*0.1
                write_row[1:] = [np.min([x+y,1.]) for x,y in zip(write_row[1:],noise_vec.tolist())]
            if isTrain:
                out = v[1]
                write_row.extend([out])
                train_data_mat[out].append(write_row)

            writer.writerow(write_row)

    if isTrain:
        with open(file_name+"_train_ordered.csv", 'w',newline='') as f2:
            writer2 = csv.writer(f2)
            writer2.writerow(header)
            for d_cls in train_data_mat:
                for r in d_cls:
                    writer2.writerow(r)

# s for single value
# v for vector
# mul for multiplication all values per id
# mul_sev for multiply the vector by severity
# mu for mean
# n for nomarlize

# removed 'sev':['v','n'],
include = {'id':['s'],'loc':['v'],'feat':['v'],'sev':['s'],'eve':['v'],'res':['v']}
file_name = 'features_2'


#write_file(file_name,train_data,feature_data,severity_data,event_data,resource_data,include,True,False)
#write_file(file_name,test_data,feature_data,severity_data,event_data,resource_data,include,False,False)
write_severity_in_order('after_comp',train_data,test_data)
#select_features('features_modified_train.csv','features_modified_test.csv',True)