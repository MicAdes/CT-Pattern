import csv
import torch
import numpy as np
import os

def write_out_results(filepath, mc, collect_data, classes):
    
    nun_digits = 5
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)

        row = ["category","object","top1 acc"]    
        if mc:
            row.append("stage1 acc")     
        row.append("pred")
        row.append("p pred")
        
        for cla in classes:
            row.append(str("p(" + cla + ")"))
        writer.writerow(row)


        for i in range(len(collect_data["obj_names"])):
            row = []
            obj = collect_data["obj_names"][i]
            
            label_index = collect_data["labels"][i]
            category = classes[label_index]
            output = collect_data["outputs"][i]
            
            row.append(category)
            row.append(obj)

            pred_prob, pred_index = torch.max(output.data, 0)
            res = 0
            if label_index == pred_index:
                res = 1
            row.append(str(res))
            
            if mc:
                mc_output = collect_data["outputs_mc"][i]
                mc_label = collect_data["labels_mc"][i]
                _, mc_pred_index = torch.max(mc_output.data, 0)
                res = 0
                if mc_label == mc_pred_index:
                    res = 1
                row.append(str(res))

            row.append(classes[pred_index])
            row.append(str(pred_prob.item()).replace(".",","))
          
            for prob in output:
                row.append(str(round(float(prob),nun_digits)).replace(".",","))
            writer.writerow(row)

def write_out_results2(filepath, collect_data, classes):   
    nun_digits = 5
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # ["category","object","top1 acc", "pred", "p pred", p(c1), ..., p(cn)]   
        row = ["category","object","top1 acc"]    
         
        row.append("pred")
        row.append("p pred")
        
        for cla in classes:
            row.append(str("p(" + cla + ")"))
        writer.writerow(row)


        for i in range(len(collect_data["img_names"])):
            row = []
            obj = collect_data["img_names"][i]
            
            label_index = collect_data["labels"][i]
            category = classes[label_index]
            output = collect_data["outputs"][i]
            
            row.append(category)
            row.append(obj)

            pred_prob, pred_index = torch.max(output.data, 0)
            res = 0
            if label_index == pred_index:
                res = 1
            row.append(str(res))
                        

            row.append(classes[pred_index])
            row.append(str(pred_prob.item()).replace(".",","))
          
            for prob in output:
                row.append(str(round(float(prob),nun_digits)).replace(".",","))
            writer.writerow(row)

def write_meta_classes(filepath, meta_classes):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["Meta classes:"])
        
        for mc in meta_classes:
                row = []
                for cl in meta_classes[mc]:
                       row.append(cl)
                writer.writerow(row)

def write_NNC_accuracies(filepath, NNC_accuracies):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["Number classifier","NNC accuracy","mean","std"])
        writer.writerow(["all","all",np.mean(np.array(NNC_accuracies), axis=0),np.std(np.array(NNC_accuracies), axis=0)])
        for i in range(len(NNC_accuracies)):
            writer.writerow([str(i+1),NNC_accuracies[i]])

def write_accuracies(filepath, accuracies):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["Number classifier","accuracy"])
        
        for i in range(1,1+len(list(accuracies.keys()))):
            writer.writerow([str(i),accuracies[str(i)]])

def write_graph_result(filepath, accuracy):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)   
        writer.writerow([str(accuracy)])
 
def write_lines(filepath, lines):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in lines: 
            writer.writerow(line)

def write_resultsAllModels(filepath, inhalt, classes, dict_order, class_depths):
    count_rights = 0
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        firstRow = ["Class","Class label"]

        for i in range(len(dict_order)):
            firstRow.append(dict_order[i] + " acc")

        firstRow.append("graph acc")

        writer.writerow(firstRow)
        
        

        for i in range(int(inhalt[0].shape[0])):
            row = []
            class_ = classes[int(inhalt[0][i])]
            row.append(class_)# class name
            row.append(inhalt[0][i])##class label
            
            img_count=0
            for k in range(int((len(inhalt)-1)/2)):
                if inhalt[1+2*k][i] == inhalt[2+2*k][i]:
                    row.append("1")
                    img_count += 1
                else: 
                    row.append("0")

            final_result = max(0,img_count - class_depths[class_])
            count_rights += final_result
            row.append(str(final_result))

            writer.writerow(row)

    acc = float(count_rights)/int(inhalt[0].shape[0])
    
    write_graph_result(os.path.split(filepath)[0] + "/results.txt", acc)

    return acc

