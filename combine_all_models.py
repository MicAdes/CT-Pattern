import torch
from torch import autograd
import argparse
import torch.backends.cudnn as cudnn
from shutil import copyfile
import time
import os
import numpy as np

from torch.utils.data import DataLoader
from custom_dataset import build_class_group, MultiViewDataSetALL
from test_model import calculate_performance
import write_functions as writer
import read_functions as reader
from DataSetup import get_bottleneck_dir, get_MN10_classes, get_MN40_classes

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='which gpu', default=0)


def get_csv_output_paths(model_paths):
    csv_files = {}

    for key in model_paths.keys():
        csv_files[key] = []
        for i in range(len(model_paths[key])):
            p = os.path.split(model_paths[key][i])[0]
            f = os.path.split(model_paths[key][i])[1]

            if "mc" in f:
                csv_files[key].append(p + "/outputs_best_graph_mc_acc.csv")
            else:
                csv_files[key].append(p + "/outputs_best_graph_acc.csv")

    return csv_files



def save_combined_architecture(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def combine(model_info_path, dataset):
    start_run = time.strftime("%b_%d_%Y_%H-%M")
    
    global args
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    model_paths,num_classifier,dict_order = reader.read_all_combine_input_csv(model_info_path)
    
    bottleneck_dir= get_bottleneck_dir()
    classes = get_MN40_classes()

    if dataset == "MN10":
        bottleneck_dir.replace("40v2png_ori4" ,"10v2png_ori4")
        classes = get_MN10_classes()
     
    meta_classes, meta_classes_idx, class_idx, index_array = build_class_group([],classes)
    dataloader =  MultiViewDataSetALL(bottleneck_dir,"test", classes, meta_classes_idx)
    
    save_dir = os.path.split(os.path.split(model_paths["main"])[0])[0] + "/MN" + str(len(classes)) + "_CombinedResults_" + start_run + "/"       
    reader.ensure_dir(save_dir) 
    copyfile(model_info_path, os.path.join(save_dir,"Model_components.csv"))

    class_depths = {x: -1 for x in classes}
    
    inhalt = []
    y = np.array(dataloader.y)
    inhalt.append(y)
    
    for step in dict_order:
        inhalt,class_depths = loadandForwardModel(inhalt, step, model_paths, num_classifier[step],class_depths,bottleneck_dir,args,device)
 
    acc = writer.write_resultsAllModels(os.path.join(save_dir,"Combined_results.csv"), inhalt, classes, dict_order,class_depths)
    
    return acc

def loadandForwardModel(inhalt, step, model_paths, num_classifier_array,class_depths, bottleneck_dir,args,device):
    mc_input, classes = reader.read_meta_classes(model_paths[step] + "/meta_classes.csv")
    meta_classes, meta_classes_idx, class_idx, index_array = build_class_group(mc_input,classes)
    
    for x in classes:
        class_depths[x] += 1

    ckpt_path = model_paths[step] + "/{}/"
    nary_ecoc_test_result = []
    y_test = []

    nary_ecoc = gen_nary_ecoc(num_class=len(meta_classes_idx), num_meta_class=len(meta_classes_idx), num_classifier=len(num_classifier_array))
    for i in range(len(num_classifier_array)):
        j = int(num_classifier_array[i])
        
        pred_labels, labels = test(args, bottleneck_dir, ckpt_path.format(j), classes, meta_classes_idx, device)
        sel = (torch.sum(pred_labels == labels.data)).double()/int(pred_labels.shape[0])
        

        nary_ecoc_test_result.append(pred_labels)
        y_test = labels

    y_test = y_test.unsqueeze(1).to("cpu").numpy()
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    inhalt.append(y_test) 
    for i in range(len(nary_ecoc_test_result)):
        nary_ecoc_test_result[i] = nary_ecoc_test_result[i].unsqueeze(1).to("cpu")
    all_pred_labels = np.concatenate(nary_ecoc_test_result, axis=1)
    
    pred_labels,acc = calculate_performance(all_pred_labels[:, 0:len(num_classifier_array)], nary_ecoc[:, 0:len(num_classifier_array)], y_test)
    inhalt.append(pred_labels)

    return inhalt,class_depths

 
def test(args, bottleneck_dir, save_dir, classes, meta_classes_idx, device):
    dataset = MultiViewDataSetALL(bottleneck_dir,"test", classes, meta_classes_idx)
    dataloader = DataLoader(dataset, batch_size=4)
    
    do_print = True
    model = []
    if len(classes) == len(meta_classes_idx):
        model = torch.load(os.path.join(save_dir,"Best_graph_acc.pth"))
    else:
        model = torch.load(os.path.join(save_dir,"Best_graph_mc_acc.pth"))
        
    model.to(device)
    cudnn.benchmark = True

    best_pred_labels = []
    cum_labels = []
    model.eval()  
    
    # Iterate over data.
    for i, (inputs, _, mc_labels, _) in enumerate(dataloader):
        inputs = inputs.to(device)
        mc_labels = mc_labels.to(device)
        with torch.set_grad_enabled(False):                                                                                       
            outputs = model(inputs)   
            outputs_mc = model.get_mc_output(outputs, meta_classes_idx)    
            _, preds = torch.max(outputs_mc.data, 1)   
            
            if i == 0:
                best_pred_labels = preds
                cum_labels = mc_labels                
            else:
                best_pred_labels = torch.cat((best_pred_labels,preds), dim = 0)            
                cum_labels = torch.cat((cum_labels,mc_labels), dim = 0)        
    
    return best_pred_labels, cum_labels    




if __name__ == '__main__':
    model_info_path = "./results/CTP1/all_models.csv"
    dataset = "MN40"
    _ = combine(model_info_path,dataset)

  