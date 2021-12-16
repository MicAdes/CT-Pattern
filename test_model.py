import torch
from torch import autograd
import argparse
import torch.backends.cudnn as cudnn
import os
import numpy as np
from torch.utils.data import DataLoader
from custom_dataset import MultiViewDataSet, build_class_group
import read_functions as reader
import write_functions as writer
from scipy.spatial.distance import hamming

parser = argparse.ArgumentParser(description='Test ConfusionTreePattern')
parser.add_argument('--gpu', help='which gpu', default=0)

def createERICodingMatrix(num_classes, num_classifier):
    a = []
    for j in range(num_classes):
            a.append([j]*num_classifier)
    
    return np.asarray(a)

def calculate_performance(preds, mc_labels, labels):
    pred_classes = []
    for i in range(preds.shape[0]):
        result = []
        for j in range(mc_labels.shape[0]):
            hamming_distance = hamming(preds[i], mc_labels[j, :])
            result.append(hamming_distance)
        label = np.argmin(np.array(result), axis=0)
        pred_classes.append(label)
    pred_classes = np.array(pred_classes)
    labels = np.array(labels).reshape(-1)
    return pred_classes, np.mean(np.equal(pred_classes, labels))



def run():
        global args
        args = parser.parse_args()
 
        save_dir = "./results/CTP1/mainMN40_cnn_graph_N5_May_24_2021_20-08/"
        bottleneck_dir= "./Bottlenecks/modelnet40v2png_ori4"
        

        mc_input,classes = reader.read_meta_classes(save_dir + "meta_classes.csv")
        meta_classes, meta_classes_idx, class_idx, index_array = build_class_group(mc_input,classes)
        
        num_classifier = 1
 
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        num_classes = len(classes)
     
        nary_ecoc_test_result = []
        y_test = []
        NNCaccuracies = []
        
        for i in range(num_classifier):
                print("\nThe {}/{} classifier:\n".format(i + 1, num_classifier))
                save_dir_i = save_dir + str(i+1) + "/"
                
                pred_labels, labels = test(args, bottleneck_dir, save_dir_i, classes, meta_classes_idx, device)
                NNCacc = (torch.sum(pred_labels == labels.data)).double()/int(pred_labels.shape[0])
                NNCaccuracies.append(float(NNCacc.item()))
      
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
                _, accuracy = calculate_performance(nary_ecoc_labels[:, 0:n], nary_ecoc[:, 0:n], y_test)
                accuracies[str(n)] = accuracy
                print("{}: {}\t{:4.2f}%".format(n, accuracy, accuracy * 100))
        
        writer.write_NNC_accuracies(save_dir + "NNCaccuracies.txt",NNCaccuracies)
        writer.write_accuracies(save_dir + "results.txt", accuracies)


def test(args, bottleneck_dir, save_dir, classes, meta_classes_idx, device):
    dataset = MultiViewDataSet(bottleneck_dir,"test", classes, meta_classes_idx)
    dataloader = DataLoader(dataset,batch_size=4)
    
    model = []
    if len(classes) == len(meta_classes_idx):
        model = torch.load(os.path.join(save_dir,"Best_graph_acc.pth"))
    else:
        model = torch.load(os.path.join(save_dir,"Best_graph_mc_acc.pth"))
    model.to(device)

    cudnn.benchmark = True

    best_pred_labels = []
    cum_labels = []
    model.eval()   # Set model to evaluate mode

    # Iterate over data.
    for i, (inputs, _, _, mc_labels, _) in enumerate(dataloader):
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

def get_labels(args, bottleneck_dir, classes):
    nary_ecoc = createERICodingMatrix(num_classes=len(meta_classes_idx), num_classifier=1)

    dataset = MultiViewDataSet(bottleneck_dir,"test", classes, nary_ecoc[:, 0]),
    dataloaders = DataLoader(dataset["test"],batch_size=4)
    
    cudnn.benchmark = True
    cum_labels = []
    
    # Iterate over data.
    for i, (_, labels) in enumerate(dataloaders["test"]):  
        if i == 0:
            cum_labels = labels                
        else:         
            cum_labels = torch.cat((cum_labels,labels), dim = 0)        

    return cum_labels

if __name__ == '__main__':
        run()