import torch
from torch import autograd
import argparse
from torchvision import transforms

from torch.utils.data import DataLoader
from architectures import FrontModel
from custom_dataset import MultiViewDataSet2
from DataSetup import get_MN10_classes
from read_functions import ensure_dir

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', help='which gpu', default = 0)
parser.add_argument('-b', '--batch-size',type=int)
parser.add_argument('--image_dir', help='Path to image directory', default= "./Images/modelnet40v2png_ori4")
parser.add_argument('--save_dir', help='Path to save created bottleneck vectors', default= "./Bottlenecks/modelnet40v2png_ori4")


def main():
    global args
    args = parser.parse_args()  

    data_dir= args.image_dir
    bottleneck_dir = args.save_dir
    bottleneck_dir_mn10 = bottleneck_dir.replace("modelnet40v","modelnet10v")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose([transforms.ToTensor(),normalize])

    dataset = {
        "train": MultiViewDataSet2(data_dir,"train", transform = data_transforms),
        "test": MultiViewDataSet2(data_dir,"test", transform = data_transforms),
    }
        
    dataloaders = {
        "train": DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(dataset["test"],batch_size=args.batch_size),
    }
    
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")   

    model = FrontModel()
    model = model.to(device)

    for phase in ["train","test"]:
        for i, (inputs, cat_name, obj_name, img_names) in enumerate(dataloaders[phase]):
            
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device)
        
            outputs = model(inputs) 
            outputs = outputs[0].to("cpu")

            path = bottleneck_dir + "/" + phase + "/" + cat_name + "/" + obj_name + "/" + img_names + "_bottleneck_alexnet.pt"
            
            ensure_dir(path)
            torch.save(outputs, path)
            if cat_name in get_MN10_classes():
                path = bottleneck_dir_mn10 + "/" + phase + "/" + cat_name + "/" + obj_name + "/" + img_names + "_bottleneck_alexnet.pt"
                ensure_dir(path)
                torch.save(outputs, path)

                 

if __name__ == '__main__':
        main()