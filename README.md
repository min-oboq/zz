# zz
---

# 챗봇 저장소

---

## 0904
* 0904 배운 것
 * import torch1
from torch1 import Torch

class PyTorch(Torch):
    def __init__(self, name):
        super().__init__(name)
        self.pyname = 'py최수길'
        
    def __str__(self):
        return '파이네임 : ' + self.pyname+ '\n이름은 :' +self.name
    
    def __eq__(self, value):
        return self.pyname == value.pyname
    
def main():
    t = Torch('최수길')
    t2 = PyTorch('최아무개')
    t3 = PyTorch('최아무개')
    t.print()
    t2.print()
    print(t2)
    # t2.sub_print()
    # print(isinstance(t, object))
    print(t2 == t3)

if __name__ == '__main__':
    main()

```python
import time
def main():
    print('this is test')
```

* 0906
* 서포터 백신
* 결정 트리
* 로지스틱 회귀와 선형