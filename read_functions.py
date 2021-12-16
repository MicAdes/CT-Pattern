import csv
import os
import numpy as np
import torch


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_output_results_csv(csv_path, classes):
    outputs = []
    labels = []

    beg = False

    with open(csv_path,newline ='\n') as fp:
        start_index = 0
        for line in fp:    
            row = line.split(";")          
            if row[0] == "category":
                for i in range(1,len(row)):
                    if row[i].startswith("p("):
                        start_index = i 
                        break
            else:
                label_ind = np.array([classes.index(row[0])])
 
                output_vec = np.array([])
                for k in range(start_index,len(row)):
                    n = np.array([float(row[k].replace(",","."))])
                    output_vec = np.append(output_vec,n)  
                             
                k = torch.from_numpy(output_vec) 
                lab = torch.from_numpy(label_ind)

                if not beg:
                    outputs = k.unsqueeze(0)
                    labels = lab
                    beg = True
                else:
                    outputs = torch.cat( (outputs,k.unsqueeze(0)), dim=0)
                    labels = torch.cat( (labels,lab), dim=0)       

    return outputs, labels

def read_preds_labels_csv(csv_path):
    classes = []
    false_labels = []
    false_preds = []

    cat_col = 0
    pred_col = 0

    with open(csv_path,newline ='\n') as fp:
        for line in fp:    
            row = line.split(";")
            start_index = 0
            
            if row[0] == "category":
                for i in range(1,len(row)):
                    if row[i].startswith("p("):
                        cat = row[i][2:len(row[i])-1]
                        if i == len(row)-1:
                            cat = cat[:len(cat)-2]                 
                        classes.append(cat)
                    if row[i] in ["pred","pred1"]:
                        pred_col = i
            else:
                label_class = classes.index(row[cat_col])
                pred_class = classes.index(row[pred_col])
                if label_class != pred_class:
                    false_labels.append(label_class)
                    false_preds.append(pred_class)

    return false_preds, false_labels, classes

def read_meta_classes(csv_path):
    meta_classes = []
    classes = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f, skipinitialspace=False,delimiter=';', quoting=csv.QUOTE_NONE)
        for row in reader:
            if "eta classes" in row[0]:
                continue  
            for c in row:
                classes.append(c)
            if len(row) > 1 :
                meta_classes.append(row)
           

    classes.sort()
    return meta_classes,classes

def read_all_combine_input_csv(csv_path):
    model_paths = {
        }
    num_classifier = {}

    dict_order = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f, skipinitialspace=False,delimiter=';', quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) > 0:
                model_paths[row[0]] = row[2]
                num_classifier[row[0]] = row[1].split(",")### num_classifier ist jetzt ein ARRAY!
                dict_order.append(row[0])
    return model_paths,num_classifier,dict_order

def read_crossFoldValidation(path):
    k = 10
    sum_false_preds = []
    sum_false_labels = []
    classes = []
    for i in range(k):
        path_i = path + "Val" + str(i+1) + "_outputs_best_graph_acc.csv"
        false_preds, false_labels, classes = read_preds_labels_csv(path_i)
        sum_false_preds += false_preds
        sum_false_labels += false_labels

    return sum_false_preds, sum_false_labels, classes