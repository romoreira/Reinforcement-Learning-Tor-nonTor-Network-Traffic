# Script to parse a PCAP and modify timestamps
# Requires Scapy
# 0.1 - 03012012
# Stephen Reese

from scapy.all import *
import sys

# Get input and output files from command line
if len(sys.argv) < 2:
        print("Usage: rewritetimestamp.py inputpcapfile")
        sys.exit(1)

# Assign variable names for input and output files
infile = sys.argv[1]

def process_packets():
    pkts = rdpcap(infile)
    cooked=[]
    timestamp = 1234567891.000000
    for p in pkts:
        p.time = timestamp
        timestamp += 0.000001
        pmod=p
        p.time
        cooked.append(pmod)

    wrpcap("out_nontor_final.pcap", cooked)

process_packets()