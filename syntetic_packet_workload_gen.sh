#!/bin/bash
sudo ip link add veth0a type veth peer name veth0b
sudo ip link set veth0a up
sudo ip link set veth0b up
sudo ifconfig veth0a 10.0.0.1 netmask 255.255.255.0
sudo ifconfig veth0b 10.0.0.2 netmask 255.255.255.0
while true
do
	sudo tcpreplay -i veth0b -tK -l $[ ( $RANDOM % 3 ) + 1 ] --loopdelay-ms $[ ( $RANDOM % 3 ) + 0 ]  dataset.pcap &
	sleep $[ ( $RANDOM % 2 ) + 1 ]s
done

#Fix pcap raw with command below:
#sudo tcpreplay-edit -i veth0a -t --efcs --enet-dmac=ce:26:e1:2d:84:92 --enet-smac=e2:5d:86:b0:cc:53 --verbose Tor.pcap
#sudo tcprewrite --infile=NonTor.pcap --dlt=enet --outfile=out_non_tor.pcap --enet-dmac=ce:26:e1:2d:84:92 --enet-smac=e2:5d:86:b0:cc:53