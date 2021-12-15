import sys
import numpy as np

def create_image(raw_packet):
    print("Raw Recebido: "+str(raw_packet))
    l = raw_packet.split(' ')
    print("New L: "+str(l))
    l = np.asarray(l)
    x = np.asmatrix(l.reshape(int(len(l)/8)+1,8))
    print(x)
#    raw_packet = np.fromstring(raw_packet, dtype=np.uint8, sep=' ')
#    print("New list from separeted spaces: "+str(raw_packet))
#    k = 8
#    for i in range(len(raw_packet)):
#        if i %k == 0:
#            sub = raw_packet[i:i+k]
#            lst = []
#            for j in sub:
#                lst.append(j)
#            print(' '.join(lst))
#    print("List: "+str(lst))
if __name__ == "__main__":
    create_image(sys.argv[1])
