# zz
---

## 0904
* 0904 배운 것

```python
import torch1
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
```

```python
import time
def main():
    print('this is test')
```

---
## 0905
* 모델 네트워크 생성
* Model 클래스의 객체 생성
* 정확도
* 재현율
* 정밀도
* F1-스코어
* 지도 학습
* K-최근접 이웃

---
## 0906
* 서포터 백신
* 결정 트리
* 로지스틱 회귀와 선형 회귀
* 비지도 학습
* K-평균 군집화
* 밀도 기반 군집 분석

---
## 0907
* AND 게이트
* OR 게이트
* XOR 게이트
* 가중치
* 바이어스
* 가중합
* 가중합 또는 전달 함수
* 활성화 함수 
* 손실함수
* 심층 신경망
* 순환 신경망
* 심층 신뢰 신경망
* 합성곱 신경망
* 풀링층
* 완전연결층
* 특성 추출 기법

---
## 09011
* 파라미터 학습 유무 지정
* 함수 생성
* 파라미터 학습 결과 옵티마이저에 전달
* 테스트 데이터를 평가 함수에 적용
* 예측 이미지 출력을 위한 전처리 함수
* 특성 맵 시각화
* 모델 객체화

## 0912
* 이미지 분류를 위한 신경망
* LeNet-5
* 데이터셋 이미지 출력
* 데이터셋 예측 결과 이미지 출력
* AleNet
* GoogLeNet