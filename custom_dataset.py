from torch.utils.data.dataset import Dataset
import os
import math
import torch
from PIL import Image


class MultiViewDataSet(Dataset):
    def build_meta_classes(self,meta_classes_idx):
        class_to_mc_idx = {}

        for i in range(len(self.classes)):
            cl = self.classes[i]

            for j in range(len(meta_classes_idx)):
                if i in meta_classes_idx[str(j)]:
                    class_to_mc_idx[cl] = j
                    break
        return class_to_mc_idx


    def __init__(self, root, data_type, classes, meta_classes_idx):
        self.x_alexnet = []
        self.y_vec = []
        self.y = []
        self.y_mc = []
        self.obj_names = []

        self.root = root
        self.classes = classes
        self.meta_classes_idx = meta_classes_idx
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.class_to_mc_idx = self.build_meta_classes(meta_classes_idx)
        
        
        # structure of files
        # root / <train/val>  / <label> / <item> / <bottleneck_file>.pt
        for label in os.listdir(root + '/' + data_type): # Label
            if not label in classes:
                continue
            for item in os.listdir(root + '/' + data_type + '/' + label):

                views_alexnet = []
                for view in os.listdir(root + '/' + data_type + '/' + label + '/' + item):   
                    if "alexnet" in view:
                        views_alexnet.append(root + '/' + data_type + '/' + label + '/' + item + '/' + view)
                    
                self.x_alexnet.append(views_alexnet)
                self.y_vec.append(self.get_one_hot_vector(self.class_to_idx[label], len(self.classes)))
                self.y.append(self.class_to_idx[label])
                self.y_mc.append(self.class_to_mc_idx[label])
                self.obj_names.append(item)

        
    
    def get_tensor_package_from_views(self, original_views):
        feature_tensor_package = []
        is_first_view = True
      
        for view in original_views:       
            feature_tensor = torch.load(view)           

            if is_first_view:
                feature_tensor_package = feature_tensor.unsqueeze(0)
                is_first_view = False                 
            else:
                feature_tensor_package = torch.cat((feature_tensor_package,feature_tensor.unsqueeze(0)), dim = 0)            

        return feature_tensor_package



    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views_alexnet = self.x_alexnet[index]     
      
        feature_tensor_package_alexnet = self.get_tensor_package_from_views(orginal_views_alexnet)
        
        return feature_tensor_package_alexnet, self.y_vec[index], self.y[index], self.y_mc[index], self.obj_names[index]
        


    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x_alexnet)


    def get_one_hot_vector(self,k,length):
        v = torch.zeros(length)
        v[k] = 1
        return v

class MultiViewDataSetCrossFoldValidation(Dataset):
    def __init__(self, valset, root, data_type, classes):
        self.x_alexnet = []
        self.y_vec = []
        self.y = []
        self.y_mc = []
        self.obj_names = []

        self.root = root
        self.classes = classes
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}        
        
        # structure of files
        # root / <train/val>  / <label> / <item> / <bottleneck_file>.pt
        for label in os.listdir(root + '/' + "train"): # Label
            if not label in classes:
                continue
            s = 0
            a = self.getTTarray(len(os.listdir(root + '/' + "train" + '/' + label)))
        
            for item in os.listdir(root + '/' + "train" + '/' + label):
                test_object = (s >= a[valset-1] and s < a[valset]) 
                bool2 = (test_object and data_type == "test") or ((not test_object) and data_type == "train")
                s += 1
                if bool2:
                    views_alexnet = []
                    for view in os.listdir(root + '/' + "train" + '/' + label + '/' + item):   
                        views_alexnet.append(root + '/' + "train" + '/' + label + '/' + item + '/' + view)
        
                    self.x_alexnet.append(views_alexnet)
                    self.y_vec.append(self.get_one_hot_vector(self.class_to_idx[label], len(self.classes)))
                    self.y.append(self.class_to_idx[label])
                    self.obj_names.append(item)
                

    def getTTarray(self,num_obj_cat):
        a = [0]
        for i in range(10):
            a.append(a[i] + math.floor(num_obj_cat/10))
            if i < int(num_obj_cat - 10*math.floor(num_obj_cat/10)):
                a[i+1] += 1
        return a
    
    def get_tensor_package_from_views(self, original_views):
        feature_tensor_package = []
        is_first_view = True
      
        for view in original_views:       
            feature_tensor = torch.load(view)           

            if is_first_view:
                feature_tensor_package = feature_tensor.unsqueeze(0)
                is_first_view = False                 
            else:
                feature_tensor_package = torch.cat((feature_tensor_package,feature_tensor.unsqueeze(0)), dim = 0)            

        return feature_tensor_package



    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views_alexnet = self.x_alexnet[index]     
        feature_tensor_package_alexnet = self.get_tensor_package_from_views(orginal_views_alexnet)
        
        return feature_tensor_package_alexnet, self.y_vec[index], self.y[index], self.obj_names[index]
        

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x_alexnet)

    def get_one_hot_vector(self,k,length):
        v = torch.zeros(length)
        v[k] = 1
        return v        

class MultiViewDataSetALL(Dataset):
    def build_meta_classes(self,meta_classes_idx):
        class_to_mc_idx = {}

        for i in range(len(self.classes)):
            cl = self.classes[i]

            for j in range(len(meta_classes_idx)):
                if i in meta_classes_idx[str(j)]:
                    class_to_mc_idx[cl] = j
                    break
        return class_to_mc_idx


    def __init__(self, root, data_type, classes, meta_classes_idx):
        self.x_alexnet = []
        self.y_vec = []
        self.y = []
        self.y_mc = []
        self.obj_names = []

        self.root = root
        self.classes = classes
        self.meta_classes_idx = meta_classes_idx
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.class_to_mc_idx = self.build_meta_classes(meta_classes_idx)
        
        
        # structure of files
        # root / <train/val>  / <label> / <item> / <bottleneck_file>.pt
        for label in os.listdir(root + '/' + data_type): # Label
            for item in os.listdir(root + '/' + data_type + '/' + label):

                views_alexnet = []
                for view in os.listdir(root + '/' + data_type + '/' + label + '/' + item):   
                    if "alexnet" in view:
                        views_alexnet.append(root + '/' + data_type + '/' + label + '/' + item + '/' + view)
                    
       
                self.x_alexnet.append(views_alexnet)
                y_ = -1
                y_mc_ = -1
                if label in classes:
                    y_ = self.class_to_idx[label]
                    y_mc_ = self.class_to_mc_idx[label]
                self.y.append(y_)
                self.y_mc.append(y_mc_)
                self.obj_names.append(item)

        
    
    def get_tensor_package_from_views(self, original_views):
        feature_tensor_package = []
        is_first_view = True
      
        for view in original_views:       
            feature_tensor = torch.load(view)           

            if is_first_view:
                feature_tensor_package = feature_tensor.unsqueeze(0)
                is_first_view = False                 
            else:
                feature_tensor_package = torch.cat((feature_tensor_package,feature_tensor.unsqueeze(0)), dim = 0)            

        return feature_tensor_package



    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views_alexnet = self.x_alexnet[index]
        
        feature_tensor_package_alexnet = self.get_tensor_package_from_views(orginal_views_alexnet)
        
        return feature_tensor_package_alexnet, self.y[index], self.y_mc[index], self.obj_names[index]
        
    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x_alexnet)


    def get_one_hot_vector(self,k,length):
        v = torch.zeros(length)
        v[k] = 1
        return v

class MultiViewDataSet2(Dataset):

    def __init__(self, root, data_type, transform=None):
        self.cat = []
        self.obj = []
        self.img_names = []
        self.img_paths = []

        self.root = root
        self.transform = transform

        # Geforderte Ordnerstruktur
        # root / <train/test>  / <label> / <item> / <view>.png
        for label in os.listdir(root + '/' + data_type): # Label
            for item in os.listdir(root + '/' + data_type + '/' + label):
                for view in os.listdir(root + '/' + data_type + '/' + label + '/' + item):                  
                    view_path = root + '/' + data_type + '/' + label + '/' + item + '/' + view
                    self.cat.append(label)
                    self.obj.append(item)
                    self.img_names.append(view)
                    self.img_paths.append(view_path)
                    

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        im = Image.open(self.img_paths[index])
        im = im.convert('RGB')
        
        if self.transform is not None:
            im = self.transform(im)              
  
        return im, self.cat[index], self.obj[index], self.img_names[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.img_names)

def build_class_group(meta_classes,classes):
    
    for j in range(len(meta_classes)):
        meta_classes[j].sort()

    mc = {}
    added_classes = []
    running_index = 0
    arr = []
    
    for i in range(len(classes)):
        if classes[i] in added_classes:
            continue
        is_in = False
        for j in range(len(meta_classes)):
            if classes[i] in meta_classes[j]:
                mc[str(running_index)] = meta_classes[j]
                for k in range(len(meta_classes[j])):
                    added_classes.append(meta_classes[j][k])
                    arr.append(classes.index(meta_classes[j][k]))
                is_in = True
        if not is_in:
            mc[str(running_index)] = [classes[i]]
            arr.append(i)

        running_index += 1

    sort_array = []
    for j in range(len(classes)):
        sort_array.append(arr.index(j))

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    mc_idx = {}
    c_idx = {}
    for i in range(len(mc)):
        mc_idx[str(i)] = []
        for cla in mc[str(i)]:
            mc_idx[str(i)].append(class_to_idx[cla])
            c_idx[str(class_to_idx[cla])] = i
                  
    return mc,mc_idx,c_idx, sort_array
    