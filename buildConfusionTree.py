import networkx as nx
import numpy as np
import argparse
import os
import time
import pickle
import read_functions as reader
from DataSetup import get_MN40_classes, get_MN10_classes
from kFoldCrossValidation import TenFoldCrossValidation


parser = argparse.ArgumentParser(description='Train ConfusionTree-pattern')
parser.add_argument('--dataset', help='which dataset', default="MN40")
parser.add_argument('--r', help="r quantile", default=0.9,type=float)
parser.add_argument('--s', help="s quantile", default=0.95,type=float)


def create_graph_matrix(preds,labels,classes):
    graph_matrix = []# init matrix with zeros
    for k in range(len(classes)):
        graph_matrix.append([0]* len(classes))

    for obj_id in range(len(preds)):
        pred_int = int(preds[obj_id])
        label_int = int(labels[obj_id])
        graph_matrix[label_int][pred_int] += 1
 
    return graph_matrix

def get_r_and_s(graph_matrix,r_quantil, s_quantil):
    real_matrix = np.zeros((len(graph_matrix),len(graph_matrix)))
    values = []
    for i in range(len(graph_matrix)):
        for j in range(len(graph_matrix)):
            if i > j:
                real_matrix[i][j] = graph_matrix[i][j]+ graph_matrix[j][i]
                values.append(real_matrix[i][j])
    values_non_zero= []
    for i in range(len(values)):
        if values[i] >0:
            values_non_zero.append(values[i])
    values_non_zero = np.array(values_non_zero)
    
    r = np.quantile(values_non_zero, r_quantil)
    s = np.quantile(values_non_zero, s_quantil)
 
    return int(round(r,0)),int(round(s,0))


def initial_graph(graph_matrix, classes, not_consider, r):
    G = nx.Graph()
    for i in range(len(graph_matrix)):
        v_1 = classes[i]
        for j in range(len(graph_matrix)):
            v_2 = classes[j]
            if v_1 in not_consider or v_2 in not_consider:
                continue
            if j > i:
                value_ij = graph_matrix[i][j] + graph_matrix[j][i]
                if value_ij > r:
                    G.add_edge(v_1, v_2, weight=value_ij)   

    return G

def cut_one_time(graph_matrix, classes, not_consider, r):         
    G = initial_graph(graph_matrix, classes, not_consider, r)
    cut_value, partition = nx.stoer_wagner(G)
    return cut_value, partition


def grouping(path,r_quantil, s_quantil):
    false_preds, false_labels, classes  = reader.read_crossFoldValidation(path)
    graph_matrix = create_graph_matrix(false_preds,false_labels,classes)

    r,s = get_r_and_s(graph_matrix,r_quantil, s_quantil)
    
    meta_classes = []
    not_consider = []

    G = initial_graph(graph_matrix, classes, not_consider, r)
    start_meta_classes = [list(list(nx.connected_components(G))[i]) for i in range(len(list(nx.connected_components(G))))]
    
    if max([len(k) for k in start_meta_classes]) <4:
        return start_meta_classes

    for i in range(len(start_meta_classes)):
        if len(start_meta_classes[i]) != max([len(k) for k in start_meta_classes]):
            not_consider += start_meta_classes[i]
            meta_classes.append(start_meta_classes[i])
  
    continue_bool = True
    while(continue_bool):
        cut_value, partition = cut_one_time(graph_matrix, classes, not_consider, r)
        
        if cut_value <= s:
            if len(partition[0])< len(partition[1]):
                not_consider += partition[0]
                meta_classes.append(partition[0])
            else:
                not_consider += partition[1]
                meta_classes.append(partition[1])
        else:
            continue_bool = False
            meta_classes.append(partition[0]+partition[1])
 
    for i in range(len(meta_classes)):
        meta_classes[i].sort() 
    meta_classes.sort()

    return meta_classes

def getMetaClasses(meta_dir, classes, state):
    path = TenFoldCrossValidation(meta_dir, classes)
    
    meta_classes = grouping(path,r_quantil, s_quantil)
    
    return meta_classes


def CTpattern(meta_dir, classes, infos, infosKeys, state, r_quantil, s_quantil):  
    meta_classes = getMetaClasses(meta_dir, classes, state)   
    infos[state] = {"classes": classes, "mc_input": meta_classes}
    infosKeys.append(state)
    if state == "main":
        state = ""
    else:
        state += "_"
    i=1
    for mc in meta_classes:
        if len(mc)>1:
            new_state = state + str(i) 
            if len(mc)<4:
                infos[new_state] = {"classes": mc, "mc_input": []}
                infosKeys.append(new_state)
                i += 1
                continue
            infos, infosKeys = CTpattern(meta_dir, mc, infos, infosKeys, new_state, r_quantil, s_quantil)
            i += 1
    return infos, infosKeys


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    global args
    args = parser.parse_args()
    
    infos = {}
    infosKeys = []
    classes = []
    if args.dataset == "MN40":
        classes = get_MN40_classes() 
    elif args.dataset == "MN10":
        classes = get_MN10_classes() 
    else:
        print("Unknown dataset. Termination")
        exit()

    state = "main"
    
    start_run = time.strftime("%b_%d_%Y_%H-%M")	
    meta_dir = os.getcwd() + "/MN" + str(len(classes)) + "_ConfusionTree_" + start_run + "/"
    reader.ensure_dir(meta_dir) 

    infos, infosKeys = CTpattern(meta_dir, classes, infos, infosKeys, state, args.r, args.s)
    
    f = {"infos": infos, "infosKeys": infosKeys}


    save_obj(f, meta_dir + "ConfusionTree")

