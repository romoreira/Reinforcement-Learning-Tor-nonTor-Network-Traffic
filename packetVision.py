import sys

def create_image(raw_packet):
    print("Raw Recebido: "+str(raw_packet))


if __name__ == "__main__":
    create_image(sys.argv[1])
