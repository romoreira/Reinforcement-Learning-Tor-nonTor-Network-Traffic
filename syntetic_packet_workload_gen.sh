#!/bin/bash
sudo ip link add veth0a type veth peer name veth0b
sudo ip link set veth0a up
sudo ip link set veth0b up
sudo ifconfig veth0a 10.0.0.1 netmask 255.255.255.0
sudo ifconfig veth0b 10.0.0.2 netmask 255.255.255.0
while true
do
	sudo tcpreplay -i veth0a -t -l $[ ( $RANDOM % 10 ) + 1 ] --preload-pcap IoT.pcap
	sleep $[ ( $RANDOM % 2 ) + 1 ]s
done
