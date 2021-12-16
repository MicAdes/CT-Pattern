import torch
import torch.nn as nn
from torch import autograd
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pickle
import time
import os
import numpy as np
from torch.utils.data import DataLoader
from custom_dataset import build_class_group, MultiViewDataSet
from architectures import Classifier
from combine_all_models import combine
import write_functions as writer
from test_model import test,createERICodingMatrix,calculate_performance

parser = argparse.ArgumentParser(description='Train ConfusionTree-pattern')
parser.add_argument('--gpu', help='which gpu', default=0)
parser.add_argument('-epochs', '--epochs', default=30,type=int)
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,help='starting learning rate')
parser.add_argument('-b', '--batch-size',default=20,type=int, help='number of 3D objects per batch')
parser.add_argument('-ct', '--confusionTree',help='path to pre-defined ConfusionTree', default= './MN40_ConfusionTree_Sep_30_2021_11-55/')


def adjust_learning_rate(optimizer, epoch, initial_lr):
    lr = initial_lr * (0.1 ** (epoch // 15))
    for parameters in optimizer.param_groups:
        parameters['lr'] = lr

def run(classes, mc_input, save_dir, key, num_classifier):
        global args
        args = parser.parse_args()
                
        start_run = time.strftime("%b_%d_%Y_%H-%M")
        bottleneck_dir= "./Bottlenecks/modelnet40v2png_ori4"
       
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

        meta_classes, meta_classes_idx, class_idx, index_array = build_class_group(mc_input,classes)
        
        print("Generating Datasets")
        dataset = {
            "train": MultiViewDataSet(bottleneck_dir,"train", classes, meta_classes_idx),
            "test": MultiViewDataSet(bottleneck_dir,"test", classes, meta_classes_idx),
        }
        print("Generating Dataloaders")
        dataloaders = {
            "train": DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True, num_workers=0),
            "test": DataLoader(dataset["test"],batch_size=args.batch_size),
        }
        
        
        num_classes = len(classes)
        save_path = save_dir + str(key) + "MN" + str(len(classes)) + "_cnn_graph_N" + str(num_classifier) + "_" + start_run + "/"
        ckpt_path = save_path + "{}/"
        ensure_dir(save_path)
        writer.write_meta_classes(os.path.join(save_path,"meta_classes.csv"), meta_classes)
        
        NNC_accuracies = []
        y_test = []
        nary_ecoc_test_result = []

        for i in range(num_classifier):
            save_dir_i =ckpt_path.format(i + 1)
            ensure_dir(save_dir_i)
            train(save_dir_i, classes, meta_classes, meta_classes_idx,index_array, device, args,dataset, dataloaders)
            
            pred_labels, labels = test(args, bottleneck_dir, save_dir_i, classes, meta_classes_idx, device)
            sel = (torch.sum(pred_labels == labels.data)).double()/int(pred_labels.shape[0])
            NNC_accuracies.append(float(sel.item()))

            nary_ecoc_test_result.append(pred_labels)
            y_test.append(labels)
        
        y_test = y_test[0].unsqueeze(1).to("cpu").numpy()
        y_test = np.reshape(y_test, (y_test.shape[0]))

        for i in range(len(nary_ecoc_test_result)):
            nary_ecoc_test_result[i] = nary_ecoc_test_result[i].unsqueeze(1).to("cpu")

        nary_ecoc_labels = np.concatenate(nary_ecoc_test_result, axis=1)
        nary_ecoc = createERICodingMatrix(len(meta_classes_idx), num_classifier)

        accuracies ={}
        nl = [l for l in range(1,1+num_classifier)] 
        for n in nl:
                _,accuracy = calculate_performance(nary_ecoc_labels[:, 0:n], nary_ecoc[:, 0:n], y_test)
                accuracies[str(n)] = accuracy
                print("{}: {}\t{:4.2f}%".format(n, accuracy, accuracy * 100))
        
        writer.write_NNC_accuracies(save_path + "NNC_accuracies.txt",NNC_accuracies)
        writer.write_accuracies(save_path + "results.txt", accuracies)


        return save_path



def train(save_path, classes, class_groups, class_groups_idx,index_array, device, args, dataset, dataloaders):   
        model = Classifier(num_classes = len(classes), class_groups = class_groups, class_groups_idx = class_groups_idx, index_array = index_array)
        model.to(device)
  
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, dampening=0, nesterov = True)
        criterion = nn.CrossEntropyLoss()
        
        since = time.time()
        OAA = (len(classes) == len(class_groups))
        best_acc = 0.0
        best_acc_epoch = 0
        best_acc_mc = 0.0
        best_acc_mc_epoch = 0
       
        cudnn.benchmark = True
 
        for epoch in range(args.epochs):
                print('-' * 10)
                print('Epoch {}/{}'.format(1+epoch, args.epochs))
                
                adjust_learning_rate(optimizer, epoch,args.lr)
                start_epoch = time.time()
                # Each epoch has a training and validation phase
                for phase in ['train', 'test']:
                        if phase == 'train':
                                model.train()  # Set model to training mode
                        else:
                                model.eval()   # Set model to evaluate mode

                        running_meta_loss = 0.0
                        running_meta_corrects = 0
                        running_meta_mc_corrects = 0
                        # Iterate over data.
                        for i, (inputs, label_vec, labels, mc_labels, obj_names) in enumerate(dataloaders[phase]):
                                inputs = inputs.to(device)
                                label_vec = label_vec.to(device)
                                labels = labels.to(device)
                                mc_labels = mc_labels.to(device)
                               
                                # zero the parameter gradients
                                optimizer.zero_grad()
                                with torch.set_grad_enabled(phase == 'train'):                                                                                       
                                        outputs = model(inputs) 
                                        outputs_mc = model.get_mc_output(outputs, class_groups_idx)

                                        _, preds = torch.max(outputs.data, 1)
                                        _, mc_preds = torch.max(outputs_mc.data,1)
                                        
                                        meta_loss = criterion(outputs, labels) # Compute Error/loss

                                        # backward + optimize only if in training phase
                                        if phase == 'train':
                                                meta_loss.backward()
                                                optimizer.step()
                                       
                                # statistics
                                running_meta_loss += meta_loss.item() * inputs.size(0)
                                running_meta_corrects += (torch.sum(preds == labels.data)).double()
                                running_meta_mc_corrects += (torch.sum(mc_preds == mc_labels.data)).double()

                                
                        
                        epoch_meta_loss = running_meta_loss / len(dataset[phase])
                        epoch_meta_acc = running_meta_corrects/ len(dataset[phase])
                        epoch_meta_acc_mc = running_meta_mc_corrects/ len(dataset[phase])


                        print('{} Loss: {:.4f} MC-Acc: {:.2f} (Acc: {:.2f})'.format(phase, epoch_meta_loss, 100*epoch_meta_acc_mc, 100*epoch_meta_acc))
                        
                        if phase == 'test':
                                if epoch_meta_acc > best_acc:
                                        best_acc = epoch_meta_acc
                                        best_acc_epoch = epoch
                                        torch.save(model, os.path.join(save_path,"Best_graph_acc.pth"))        
                                if not OAA and epoch_meta_acc_mc > best_acc_mc:
                                        best_acc_mc = epoch_meta_acc_mc
                                        best_acc_mc_epoch = epoch
                                        torch.save(model, os.path.join(save_path,"Best_graph_mc_acc.pth"))
                                        
                                
                                print("Current record accuracy: {:.2f} in epoch {}".format(100*best_acc, 1+best_acc_epoch))        
                                if not OAA:
                                        print("Current record mc-accuracy: {:.2f} in epoch {}".format(100*best_acc_mc, 1+best_acc_mc_epoch))

                epoch_duration = time.time()-start_epoch
                print("Time needed for epoch ", 1+epoch, ": ", epoch_duration)
                
        print("")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))


        # load, evaluate and save results best network
        model = torch.load(os.path.join(save_path,"Best_graph_acc.pth"))
        collect_data = evaluate_and_collect_data(model, dataset["test"], dataloaders["test"], class_groups_idx, device)
        writer.write_out_results(os.path.join(save_path,"outputs_best_graph_acc.csv"), False, collect_data, classes)

        if not OAA:
                model = torch.load(os.path.join(save_path,"Best_graph_mc_acc.pth"))
                collect_data = evaluate_and_collect_data(model, dataset["test"], dataloaders["test"], class_groups_idx, device)
                writer.write_out_results(os.path.join(save_path,"outputs_best_graph_mc_acc.csv"), True, collect_data, classes)




def evaluate_and_collect_data(model, dataset, dataloader, class_groups_idx, device):
        collect_data = {
                "outputs":[],
                "outputs_mc": [],
                "labels":[],
                "labels_mc": [],
                "obj_names": []
        }

        # Iterate over data.
        for i, (inputs, _, labels, mc_labels, obj_names) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                mc_labels = mc_labels.to(device)    

                with torch.set_grad_enabled(False):                                                                                       
                        outputs = model(inputs)  
                        outputs_mc = model.get_mc_output(outputs, class_groups_idx)      
                        _, preds = torch.max(outputs.data, 1)
                        _, mc_preds = torch.max(outputs_mc.data,1)

                if i == 0:
                        collect_data["outputs"] = outputs
                        collect_data["outputs_mc"] = outputs_mc
                        collect_data["labels"] = labels
                        collect_data["labels_mc"] = mc_labels  
                else:        
                        collect_data["outputs"] = torch.cat((collect_data["outputs"], outputs), dim=0)
                        collect_data["outputs_mc"] = torch.cat((collect_data["outputs_mc"],outputs_mc), dim=0)
                        collect_data["labels"] = torch.cat((collect_data["labels"], labels), dim=0)
                        collect_data["labels_mc"] = torch.cat((collect_data["labels_mc"], mc_labels), dim=0)

                for obj_name in obj_names:  
                        collect_data["obj_names"].append(obj_name) 

        return collect_data


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



if __name__ == '__main__':
    global args
    args = parser.parse_args()
    pathConfusionTree = args.ct
    confusionTreeFile = load_obj(pathConfusionTree + "ConfusionTree")
    
    num_class = {}
    for key in confusionTreeFile["infosKeys"]:
        num_class[key] = 1

    meta_save_dir = "./results/CTP1/"
    ensure_dir(meta_save_dir) 
    graph_model_paths = []
    for key in confusionTreeFile["infosKeys"]: 
        model_path = run(confusionTreeFile["infos"][key]["classes"],confusionTreeFile["infos"][key]["mc_input"], meta_save_dir, key, num_class[key])
        line = [key,1,model_path]
        graph_model_paths.append(line)
    
    writer.write_lines(os.path.join(meta_save_dir,"all_models.csv"),graph_model_paths)

    acc = combine(os.path.join(meta_save_dir,"all_models.csv"),"MN40")
    print("CTP Accuracy :", acc)