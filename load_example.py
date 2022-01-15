import torch
from torchvision.transforms import transforms
from PIL import Image
from torchvision import datasets, models, transforms
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import sys
from csv import writer
import threading
import time


SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print("CPU-based classification")

time_list = []

know_classes = ['NonTor', 'Tor']

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Inicializando cada variável específica para cada modelo
    model_ft = None
    input_size = 0

    if model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "mobilenet":
      model_ft = models.mobilenet_v2(pretrained=use_pretrained)
      set_parameter_requires_grad(model_ft, feature_extract)
      num_ftrs = model_ft.classifier[1].in_features
      model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
      input_size = 224

    elif model_name == "resnet":
      model_ft = models.resnet18(pretrained=use_pretrained)
      set_parameter_requires_grad(model_ft, feature_extract)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs, num_classes)
      input_size = 224

    elif model_name == "alexnet":
      model_ft = models.alexnet(pretrained=use_pretrained)
      set_parameter_requires_grad(model_ft, feature_extract)
      num_ftrs = model_ft.classifier[6].in_features
      model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
      input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = True


def cnn_start():
    model = 0
    input_size = 0
    model_name = "densenet"
    model, input_size = initialize_model(model_name, num_classes=2, feature_extract=True, use_pretrained=True)

    checkpoint = torch.load(Path('/home/rodrigo/PycharmProjects/adaptative-monitoring/models_trained/densenet-packetvision.pth'))
    model.load_state_dict(checkpoint)
    model.eval()

    return model

def get_cnn_complexity(model):
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def write_csv(register):
    with open(str(sys.argv[2])+'_exp_time_spent_on_prediction.csv', 'a') as f:
        writer_object = writer(f)
        writer_object.writerow(register)
        f.close()
        #print("Prediction time recorded")

def cnn_predict(image_name):
    model = cnn_start()

    test_transforms = transforms.Compose([
        transforms.Resize(size=[224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    path = Path('/home/rodrigo/PycharmProject/adaptative-monitoring/tmp_pooling/'+str(image_name))

    
    image = Image.open(path)

    input = test_transforms(image)
    input = torch.unsqueeze(input, 0)


    output = model(input)
    

    prediction = output.max(1, keepdim=True)[1]

    #print("Prediction: "+str(know_classes[int(prediction.item())]))
    return know_classes[int(prediction.item())]

if __name__ == '__main__':
    print(cnn_predict(sys.argv[1]))
