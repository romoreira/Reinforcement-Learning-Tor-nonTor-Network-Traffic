import os
import sys
from scapy.all import *


def runner(pkt_amount, duration, interface):
    cmd = 'sudo dumpcap -i '+str(interface)+' -c '+str(pkt_amount)+' -a duration:'+str(duration)+ ' -w /tmp/output.pcap'
    os.system(cmd)

if __name__ == "__main__":
   runner(sys.argv[1], sys.argv[2], sys.argv[3])
   packets = rdpcap('output.pcap')
   for i in range(len(packets)):
    print(raw(packets[i]))

