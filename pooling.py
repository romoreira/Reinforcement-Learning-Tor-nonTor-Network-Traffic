from __future__ import division
import os
import sys
from scapy.all import *
from scapy.all import bytes_hex
from scapy.all import raw
from scapy.all import hexdump
import calendar
import time
import subprocess
from load_example import cnn_predict

current_network_status = 0

def runner(pkt_amount, duration, interface):
    cmd = 'sudo dumpcap -i '+str(interface)+' -c '+str(pkt_amount)+' -a duration:'+str(duration)+ ' -w /tmp/output.pcap'
    os.system(cmd)

def main(pkt_amount, duration, interface_name):
    current_network_status = 0
    runner(int(pkt_amount), int(duration), interface_name)
    for i in range(int(pkt_amount)):
        packets = rdpcap('/tmp/output.pcap')

    for i in range(len(packets)):
        cmd = "python3 packetVision.py '"+str(linehexdump(packets[i], onlyhex=1, dump=True))+"' "+str(calendar.timegm(time.gmtime()))+" "+str(i)
        os.system(cmd)
    cmd = 'sudo rm /tmp/output.pcap'
    os.system(cmd)

    print("End of pooling")


    path, dirs, files = next(os.walk("/home/rodrigo/adaptative-monitoring/tmp_pooling/"))
    returned_value = ''
    for x in os.listdir("/home/rodrigo/adaptative-monitoring/tmp_pooling/"):
           if x.endswith(".png"):
               #cmd = 'python3 load_example.py '+str(x)
            returned_value = cnn_predict(x)
            if returned_value == 'iot':
                   current_network_status = current_network_status + 1

    #print("IoT Traffic Percent on the Network: "+str("{0:.0f}%".format(current_network_status/len(files) * 100)))
    return (current_network_status/len(files) * 100)

if __name__ == "__main__":
   runner(int(sys.argv[1]), sys.argv[2], sys.argv[3])
   for i in range(int(sys.argv[1])):
       packets = rdpcap('/tmp/output.pcap')

   for i in range(len(packets)):
    cmd = "python3 packetVision.py '"+str(linehexdump(packets[i], onlyhex=1, dump=True))+"' "+str(calendar.timegm(time.gmtime()))+" "+str(i)
    os.system(cmd)
   cmd = 'sudo rm /tmp/output.pcap'
   os.system(cmd)
   print("End of pooling")

    
   path, dirs, files = next(os.walk("/home/rodrigo/adaptative-monitoring/tmp_pooling/"))
   returned_value = ''
   for x in os.listdir("/home/rodrigo/adaptative-monitoring/tmp_pooling/"):
       if x.endswith(".png"):
           #cmd = 'python3 load_example.py '+str(x)
           returned_value = cnn_predict(x)
           if returned_value == 'iot':
               current_network_status = current_network_status + 1

   print("IoT Traffic Percent on the Network: "+str("{0:.0f}%".format(current_network_status/len(files) * 100)))
    

