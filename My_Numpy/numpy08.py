import numpy as np
from numpy03 import npprint

def main():
    a1 = np.fromfunction(lambda x, y, z: x + y + z, (2, 5, 4), dtype=np.int8)
    npprint(a1)
    # a2 = a1[:, 1::2, :3]
    a2 = a1[:][1::2][:3]
    npprint(a2)
    

if __name__ == "__main__":
    main()