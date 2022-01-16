# Adaptative Monitoring for Online Prediction of Tor/nonTor Traffic.
### Requirements:

* ```sudo apt-get install libpcap-dev```
* ```sudo apt-get install wireshark```
* ```sudo apt-get install tshark```
* ```sudo apt-get update```
* ```sudo apt-get install  wirehark-common```
* ```sudo apt-get install python3-dev python3-pip```
* ```sudo pip3 install --pre scapy[complete]```
* ```sudo pip3 install numpy```
* ```sudo pip3 install pandas```
* ```sudo pip3 install Pillow```

### How to Run:

## ```sudo python3 pooling.py 1 1 eth0``` 
Where first parameter refers to amount of packet capture, second refers to time of capture and last one refers to interface to be monitored.

---
# Reinforcement Learning Requirements Steps
## Ubuntu 18.04
#### This steps below refers to execution of Reinforcement Learning combined with [Packet Vision](https://romoreira.github.io/packetvision/) to classify Tor nonTor traffic. To customize this enviromnet to another, please refer to this [page](https://github.com/romoreira/adaptative-monitoring/tree/main/gym-basic).
    sudo apt-get update
    git clone https://github.com/romoreira/adaptative-monitoring.git
    sudo apt-get install python3.7
    which python3.6
    which python3.7
    update-alternatives --list python3
    python3
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 3
    update-alternatives --list python3
    update-alternatives --config python3
    pip3 install gym
    sudo apt-get install cython
    pip3 install cython
    pip3 install numpy
    pip3 install --upgrade pip3
    pip3 install --upgrade pip
    sudo apt-get install gofortran
    sudo apt-get install python3-dev
    sudo apt-get install gcc
    pip3 install gyn
    pip3 install gym
    sudo apt-get install python3.7-dev
    pip3 install stable-baselines[mpi]
    pip3 install tensorflow==1.15.0
    sudo apt-get install python3-opencv
    sudo apt-get install cmake
    pip3 install opencv-python
    sudo apt install libopenmpi-dev
    pip3 install ale-py



Finnaly, rum

{}