from numba import jit

@jit(nopython=True)
def convert_packet_to_int(packet_hex):
    for i in range(len(packet_hex)):
        for j in range(8):
            packet_hex[i][j] = int(packet_hex[i][j], 16)
    return packet_hex

@jit(nopython=True)
def expand_pixels_image(packet_hex, data):
    for i in range(len(packet_hex)):
        for j in range(8):
            data[i][j] = [data[i][j], data[i][j], data[i][j]]
