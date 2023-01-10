# Adaptive Network Monitoring for Online Classification of Tor/*non*Tor Traffic near to real-time.
### Requirements:

* ```conda create --name rl-cnn python=3.7 --file requirements.txt```
* ```conda activate rl-cnn```
* ```sudo apt-get install libopenmpi-dev```
* ```sudo apt-get install python3-scapy```
* ```cd gym-basic```
* ```python setup.py install``` You should proceed with this steps in each machine where you want to run. To register the environment.

### :arrow_right: Need to run dataset traffic before any task.
The dataset was build from CiC Univsersity of New Brunswich (Canadian) available (here)[https://www.unb.ca/cic/datasets/index.html].
This dataset contains all *non*-Tor classes such as audio, browsing, chat, email, p2p, transfer, video and voip. We merged all data of these classes in a single dataset named *non*-Tor.pcap. 
After, we used ISCXTor2016 as a Tor Traffic. 

#### Run syntetic Traffic generator based on Tor/*non*Tor Dataset
    ./syntetic_packet_workload_gen.sh

### :arrow_right: Interface pooling test:

    sudo python3 pooling.py 1 1 eth0
Where first parameter refers to amount of packet capture, second refers to time of capture and last one refers to interface to be monitored.
```Note that pooling command should run as root. Hence, install torch and its dependences as root accordingly```

### :arrow_right: Finnaly, run: :on:

     python3 teste.py --gamma 0.9 --env "gym_basic:basic-v1" --n-episode 200 --batch-size 1 --hidden-dim 12 --capacity 1000 --max-episode 50 --min-eps 0.01

