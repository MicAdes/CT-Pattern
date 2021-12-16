import torch
import torch.nn as nn
from torch import autograd
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import os
from custom_dataset import MultiViewDataSetCrossFoldValidation
from architectures import Classifier
import write_functions as writer
from dataSetup import get_MN10_classes, get_MN40_classes


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='which gpu', default = 0)
parser.add_argument('-epochs', '--epochs', default=10, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='starting learning rate')
parser.add_argument('-b', '--batch-size', default=20,type=int, help='number of 3D objects per batch')



def run(valset, save_dir, classes):
        global args
        args = parser.parse_args()
        bottleneck_dir= "./Bottlenecks/modelnet40v2png_ori4"
        
        start_run = time.strftime("%b_%d_%Y_%H-%M")
        
       	dataset = {
            "train": MultiViewDataSetCrossFoldValidation(valset,bottleneck_dir,"train", classes),
            "test": MultiViewDataSetCrossFoldValidation(valset,bottleneck_dir,"test", classes),
        }
        
        dataloaders = {
            "train": DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True, num_workers=0),
            "test": DataLoader(dataset["test"],batch_size=args.batch_size),
        }
   
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        num_classes = len(classes)

        model = Classifier(num_classes = num_classes, class_groups = [] ,class_groups_idx = [], index_array = [])
        model.to(device)
  
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, dampening=0, nesterov = True)
        criterion = nn.CrossEntropyLoss()
       
        best_acc = 0.0
        best_acc_epoch = 0
     
        cudnn.benchmark = True

        for epoch in range(args.epochs):
                print('-' * 10)
                print('Epoch {}/{}'.format(1+epoch, args.epochs))
                
                start_epoch = time.time()
                # Each epoch has a training and validation phase
                for phase in ['train', 'test']:
                        if phase == 'train':
                                model.train()  # Set model to training mode
                        else:
                                model.eval()   # Set model to evaluate mode

                        running_meta_loss = 0.0
                        running_meta_corrects = 0
                       
                        # Iterate over data.
                        for i, (inputs, _, labels, obj_names) in enumerate(dataloaders[phase]):
                                inputs = inputs.to(device)
                                labels = labels.to(device)
                                
                                # zero the parameter gradients
                                optimizer.zero_grad()
                                with torch.set_grad_enabled(phase == 'train'):                                                                                       
                                        outputs = model(inputs)                                        
                                        _, preds = torch.max(outputs.data, 1)
                                        meta_loss = criterion(outputs, labels) # Compute Error/loss
                                        if phase == 'train':
                                                meta_loss.backward()
                                                optimizer.step()
                                # statistics
                                running_meta_loss += meta_loss.item() * inputs.size(0)
                                running_meta_corrects += (torch.sum(preds == labels.data)).double()
 
                        epoch_meta_loss = running_meta_loss / len(dataset[phase])
                        epoch_meta_acc = running_meta_corrects/ len(dataset[phase])
                        
                        print('{} Loss: {:.4f} Acc: {:.2f}'.format(phase, epoch_meta_loss, 100*epoch_meta_acc))
                        
                        if phase == 'test':
                                if epoch_meta_acc > best_acc:
                                        best_acc = epoch_meta_acc
                                        best_acc_epoch = epoch
                                                                   
                                print("Current record accuracy: {:.2f} in epoch {}".format(100*best_acc, 1+best_acc_epoch))        
                              
        print("")   
        print('Best test Acc: {:4f}'.format(best_acc))


        collect_data = evaluate_and_collect_data(model, dataset["test"], dataloaders["test"], device)
        writer.write_out_results2(os.path.join(save_dir,"Val" + str(valset) + "_outputs_best_graph_acc.csv"), collect_data, classes)

def evaluate_and_collect_data(model, dataset, dataloader, device):
        collect_data = {
                "outputs":[],
                "labels":[],
                "img_names": []
        }

        # Iterate over data.
        for i, (inputs, _, labels, img_names) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)  

                with torch.set_grad_enabled(False):                                                                                       
                        outputs = model(inputs)      
                        _, preds = torch.max(outputs.data, 1)

                if i == 0:
                        collect_data["outputs"] = outputs
                        collect_data["labels"] = labels
                else:        
                        collect_data["outputs"] = torch.cat((collect_data["outputs"], outputs), dim=0)
                        collect_data["labels"] = torch.cat((collect_data["labels"], labels), dim=0)

                for img_name in img_names:  
                        collect_data["img_names"].append(img_name) 

        return collect_data

def TenFoldCrossValidation(meta_dir, classes):
        start_run = time.strftime("%b_%d_%Y_%H-%M")	
        save_dir = meta_dir + "MN" + str(len(classes)) + "_10crossFoldValidation_" + start_run + "/"
        ensure_dir(save_dir)
        for i in range(1,11):        
                run(i,save_dir,classes)
        return save_dir
        


if __name__ == '__main__':
        classes = get_MN40_classes() # For MN10: get_MN40_classes()
        start_run = time.strftime("%b_%d_%Y_%H-%M")	
        save_dir = os.getcwd() + "/MN" + str(len(classes)) + "_10crossFoldValidation_" + start_run + "/"
        ensure_dir(save_dir) 

        _ = TenFoldCrossValidation(save_dir,classes)

        