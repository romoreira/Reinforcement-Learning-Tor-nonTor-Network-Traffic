import os
import sys
from scapy.all import *
from scapy.all import bytes_hex
from scapy.all import raw
from scapy.all import hexdump

def runner(pkt_amount, duration, interface):
    cmd = 'sudo rm /tmp/output.pcap'
    os.system(cmd)
    cmd = 'sudo dumpcap -i '+str(interface)+' -c '+str(pkt_amount)+' -a duration:'+str(duration)+ ' -w /tmp/output.pcap'
    os.system(cmd)
    #print(os.system(cmd))

if __name__ == "__main__":
   runner(sys.argv[1], sys.argv[2], sys.argv[3])
   packets = rdpcap('/tmp/output.pcap')
   a = hexdump(packets[0], dump=True)
   print("Printing a: "+str(a))
   b = linehexdump(packets[0], onlyhex=1, dump=True)
   print("Printing b: "+str(b))
#   for i in range(len(packets)):
#    print("passing: "+str(hexdump(packets[i])))
#    cmd = "python3 packetVision.py '"+str(hexdump(packets[i]))+"'"
#    print("Command to run: "+str(cmd))
#    os.system(cmd)
#   print("End of pooling")
