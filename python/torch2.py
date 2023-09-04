import torch1
from torch1 import Torch

class PyTorch(Torch):
    def __init__(self, name):
        super().__init__(name)
        self.pyname = 'py최수길'
        
    def sub_print(self):
        print('파이네임 : ' + self.pyname+ '\n이름은 :' +slef.name)
    
def main():
    t = Torch('최수길')
    t2 = PyTorch('최아무개')
    t.print()
    t2.print()
    t2.sub_print()

if __name__ == '__main__':
    main()