import sys
import numpy as np
import pandas as pd
from PIL import Image

def create_image(raw_packet, time_stamp, pkt_number):
    l = raw_packet.split(' ')
    j = 0
    packet_hex = []
    sai = 0
    for i in range(0, len(l), 8):
        lst = []
        j = i
        while j < i + 8:
            if not((i+8) > len(l)):
                lst.append(l[j])
                j = j + 1
                #print("lst: "+str(lst))
            #print(i)
            if len(l) - i < 8:
                   j = i
                   lst = []
                   while j < len(l):
                       #print("Valor de J: "+str(j))
                       lst.append(l[j])
                       #print(lst)                    
                       #packet_hex.append(lst)
                       j = j + 1
                   packet_hex.append(lst)
                   sai = 1
            if sai == 1:
                break
        if sai == 1:
            break

        packet_hex.append(lst)
    #print("LST: "+str(packet_hex))

    if len(packet_hex[-1]) < 8:
        last_small_list = packet_hex[-1]
        i = len(packet_hex[-1])
        while i < 8:
            last_small_list.append('FF')
            i = i + 1

    #print(packet_hex)
    
    
    for i in range(len(packet_hex)):
        for j in range(8):
#            print(str(packet_hex[i][j]))
#            print("Conversao: "+str(int(packet_hex[i][j],16)))
            #print("Subistituindo: "+str(packet_hex[i][j])+ " por : "+str(int(packet_hex[i][j],16)))
            packet_hex[i][j] = int(packet_hex[i][j],16)

    #print(packet_hex)

    numeros = np.matrix(packet_hex)
    numeros = numeros.astype(int)


    #print(numeros)


    dataFrame = pd.DataFrame(numeros)
    data = dataFrame.to_numpy()

    data = data.tolist()
    #print(data[0][7])

    for i in range(len(packet_hex)):
        for j in range(8):
            data[i][j] = [data[i][j],data[i][j],data[i][j]]

    data = np.array(data)
    #print(data)


    img = Image.fromarray(data.astype('uint8'), 'RGB')
    #size=n*8
    #arr = np.zeros((size,size,3))
    #arr[:,:,0] = [[255]*size]*size
    #arr[:,:,1] = [[255]*size]*size
    #arr[:,:,2] = [[0]*size]*size
    #img = Image.fromarray(arr.astype('uint8'), 'RGB')

#    print("\nPronto pra salvar: " + str(n))
    img.save(str(time_stamp)+"_"+str(pkt_number)+"_sample.png")
    return








#    print("New L: "+str(l))
#    l = np.asarray(l)
#    x = np.asmatrix(l.reshape(int(len(l)/8)+1,8))
#    print(x)
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
    create_image(sys.argv[1], sys.argv[2], sys.argv[3])
