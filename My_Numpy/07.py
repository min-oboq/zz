import numpy as np
from numpy03 import npprint


def main():
    a1 = np.arange(1, 10 + 1) ** 2
    a2 = a1[2:9]
    npprint(a1, a2)
    a1[3] = a1[1] + a1[2]
    npprint(a1, a2)
    a2[:5:2] = 10_000
    
    for i in range(len [a1]):
        print(a1[(i + 1) * -1], end=", ")
        print()


if __name__ == "__main__":
    main()