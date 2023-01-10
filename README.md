# Adaptative Monitoring for Online Prediction of Tor/nonTor Traffic in vivo.
### Requirements:

* ```conda create --name rl-cnn python=3.7 --file requirements.txt```
* ```conda activate rl-cnn```
* ```sudo apt-get install libopenmpi-dev```
* ```cd gym-basic```
* ```python setup.py install```
* ```sudo apt-get install python3-scapy```

### Interface pooling test:

## ```sudo python3 pooling.py 1 1 eth0``` 
Where first parameter refers to amount of packet capture, second refers to time of capture and last one refers to interface to be monitored.


#### Finnaly, run:

     python3 teste.py --gamma 0.9 --env "gym_basic:basic-v1" --n-episode 200 --batch-size 1 --hidden-dim 12 --capacity 1000 --max-episode 50 --min-eps 0.01
