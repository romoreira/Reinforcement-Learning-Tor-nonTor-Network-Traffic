'''
Auhor: Rodrigo Moreira rodrigo at ufv dot br
Based on: https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952
'''

from __future__ import division
import gym
from stable_baselines.common.env_checker import check_env
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import matplotlib.pyplot as plt
import random
from gym.spaces import Discrete
from gym.spaces import Box

import sys
# insert at 1, 0 is the script path (or '' in REPL)


import os
import sys
from scapy.all import *
from scapy.all import bytes_hex
from scapy.all import raw
from scapy.all import hexdump
import calendar
import time
import subprocess
current_network_status = 0
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

from numba_functions import convert_packet_to_int
from numba_functions import expand_pixels_image

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("CPU-based classification")
time_list = []
know_classes = ['bittorrent', 'browsing', 'dns', 'iot', 'rdp', 'ssh', 'voip']
import sys
import numpy as np
import pandas as pd
from PIL import Image


class BasicEnv(gym.Env):

    def __init__(self):
        #print("Creating environment Adaptative Sampling")
        self.action_space = Discrete(100, 1)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 3 + random.randint(-3,3)
        #print("\ninit: "+str(self.state)+"\n")
        self.pooling_times = 4
        self.model = self.cnn_start()

    def step(self, action):
        print("\nStep Action Required: "+str(action))
        self.state = self.main(1 if action == 0 else action, 3330, 'wlp3s0')#1 if rando return 0, else action otherwise
        print("\nNew State after pooling: "+str(self.state))
        self.pooling_times -= 1

        print("pooling times: "+str(self.pooling_times))

        if self.state >= 93:#If IoT sampling is bigger than 90% that is correct
            reward = 1
        else:
            reward = -1

        if self.pooling_times <= 0:
            done = True
            self.pooling_times = 4
        else:
            done = False

        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.state = 3 + random.randint(-3,3)
        self.pooling_times = 4
        return self.state

    def render(slef, mode='human'):

        if mode == 'human':
            plt.imshow(np.asarray(im))
            plt.axis('off')
        elif mode == 'rgb_array':
            return np.asarray(im)



    def create_image(self, raw_packet, time_stamp, pkt_number):
        l = raw_packet.split(' ')
        j = 0
        packet_hex = []
        sai = 0
        for i in range(0, len(l), 8):
            lst = []
            j = i
            while j < i + 8:
                if not ((i + 8) > len(l)):
                    lst.append(l[j])
                    j = j + 1
                    # print("lst: "+str(lst))
                # print(i)
                if len(l) - i < 8:
                    j = i
                    lst = []
                    while j < len(l):
                        # print("Valor de J: "+str(j))
                        lst.append(l[j])
                        # print(lst)
                        # packet_hex.append(lst)
                        j = j + 1
                    packet_hex.append(lst)
                    sai = 1
                if sai == 1:
                    break
            if sai == 1:
                break

            packet_hex.append(lst)
        # print("LST: "+str(packet_hex))

        if len(packet_hex[-1]) < 8:
            last_small_list = packet_hex[-1]
            i = len(packet_hex[-1])
            while i < 8:
                last_small_list.append('FF')
                i = i + 1

        # print(packet_hex)



        #packet_hex = convert_packet_to_int(packet_hex)
        for i in range(len(packet_hex)):
            for j in range(8):
                 packet_hex[i][j] = int(packet_hex[i][j], 16)



        numeros = np.matrix(packet_hex)
        numeros = numeros.astype(int)



        dataFrame = pd.DataFrame(numeros)
        data = dataFrame.to_numpy()

        data = data.tolist()

        #data = expand_pixels_image(packet_hex, data)
        for i in range(len(packet_hex)):
            for j in range(8):
                data[i][j] = [data[i][j], data[i][j], data[i][j]]

        data = np.array(data)
        # print(data)

        img = Image.fromarray(data.astype('uint8'), 'RGB')
        # size=n*8
        # arr = np.zeros((size,size,3))
        # arr[:,:,0] = [[255]*size]*size
        # arr[:,:,1] = [[255]*size]*size
        # arr[:,:,2] = [[0]*size]*size
        # img = Image.fromarray(arr.astype('uint8'), 'RGB')

        #    print("\nPronto pra salvar: " + str(n))
        img.save(
            "/home/rodrigo/PycharmProjects/adaptative-monitoring/tmp_pooling/" + str(time_stamp) + "_" + str(
                pkt_number) + "_sample.png")
        return

    # -----------------------------------------------------------------------------------------------------------------------

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained):
        # Inicializando cada variável específica para cada modelo
        model_ft = None
        input_size = 0

        if model_name == "squeezenet":
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "mobilenet":
            model_ft = models.mobilenet_v2(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "resnet":
            model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = True

    def cnn_start(self):
        input_size = 0
        model_name = "squeezenet"
        self.model, input_size = self.initialize_model(model_name, 7, True, True)

        checkpoint = torch.load(Path('/home/rodrigo/PycharmProjects/adaptative-monitoring/models_trained/squeezenet.pth'))
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        return self.model

    def get_cnn_complexity(model):
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
                                                 verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    def write_csv(register):
        with open(str(sys.argv[2]) + '_exp_time_spent_on_prediction.csv', 'a') as f:
            writer_object = writer(f)
            writer_object.writerow(register)
            f.close()
            # print("Prediction time recorded")

    def cnn_predict(self, image_name):

        test_transforms = transforms.Compose([
            transforms.Resize(size=[224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        path = Path('/home/rodrigo/PycharmProjects/adaptative-monitoring/tmp_pooling/' + str(image_name))

        image = Image.open(path)

        input = test_transforms(image)
        input = torch.unsqueeze(input, 0)

        output = self.model(input)

        prediction = output.max(1, keepdim=True)[1]

        # print("Prediction: "+str(know_classes[int(prediction.item())]))
        return know_classes[int(prediction.item())]

    def runner(self, pkt_amount, duration, interface):
        cmd = 'dumpcap -i ' + str(interface) + ' -c ' + str(pkt_amount) + ' -a duration:' + str(
            duration) + ' -w /tmp/output.pcap'
        os.system(cmd)

    def main(self, pkt_amount, duration, interface_name):
        current_network_status = 0
        self.runner(int(pkt_amount), int(duration), interface_name)
        packets = PcapReader('/tmp/output.pcap').read_all()

        for i in range(len(packets)):
            self.create_image(str(linehexdump(packets[i], onlyhex=1, dump=True)), str(calendar.timegm(time.gmtime())), str(i))
        
        cmd = 'sudo rm /tmp/output.pcap'
        os.system(cmd)

        print("End of pooling")

        path, dirs, files = next(os.walk("/home/rodrigo/PycharmProjects/adaptative-monitoring/tmp_pooling/"))
        returned_value = ''
        for x in os.listdir("/home/rodrigo/PycharmProjects/adaptative-monitoring/tmp_pooling/"):
            if x.endswith(".png"):
                # cmd = 'python3 load_example.py '+str(x)
                returned_value = self.cnn_predict(x)
                if returned_value == 'iot':
                    current_network_status = current_network_status + 1

        cmd = 'sudo rm /home/rodrigo/PycharmProjects/adaptative-monitoring/tmp_pooling/*.png'
        os.system(cmd)

        # print("IoT Traffic Percent on the Network: "+str("{0:.0f}%".format(current_network_status/len(files) * 100)))
        return (current_network_status / len(files) * 100)


