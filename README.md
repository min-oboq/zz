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

---
## 0912
* 이미지 분류를 위한 신경망
* LeNet-5
* 데이터셋 이미지 출력
* 데이터셋 예측 결과 이미지 출력
* AleNet
* GoogLeNet
* 객체 인식을 위한 신경망
* R-CNN
* 공간 피라미드 풀링
* 이미지 분할을 위한 신경망
* 완전 합성곱 네트워크

---
## 0913
* 파이썬 플라스크 소개
* 장고
* 보틀
* FastAPI
* MVT(Model, View, Template) 모델
* flask run 명령어 실행하기
* 디버그 모드
* Rule에 변수 지정하기
* 쿠키
* 세션
* 응답

---
## 0914
* 데이터베이스를 이용한 앱 만들기
* CRUD 앱의 모듈 작성하기
* Blueprint
* 데이터베이스 조작하기
* 데이터베이스를 사용한 CRUD 앱 만들기
* 템플릿의 공통화와 상속
* config 설정하기

---
## 0918
* 사용자 인증 기능 만들기
* 앱에 인증 기능 등록하기
* 회원가입 기능 만들기
* 로그인 기능 만들기
* 로그아웃 기능 만들기
* 물체 감지 앱 등록하기
* 이미지 일람 화면 만들기
* Userlmage 모델 작성하기
* 이미지 일람 화면의 엔드포인트 만들기
* 회원가입과 로그인 화면 만들기
* 로그인 화면의 템플릿 갱신하기

---
## 0925
* 성능 최적화
* 조기 종료를 이용한 성능 최적화
* 자연어 전처리
* MaxAbsScaler()
* 정규화
* 표제어 추출
* 불용어 제거

---
## 0926
* BCELoss 손실 함수
* 시그모이드 함수
* BCEWithLogitsLoss 손실 함수
* 희소 표현 기반 임베딩
* 원-핫 인코딩
* 횟수 기반 임베딩
* TF-IDF
* 예측 기반 임베딩
* 트랜스포머 어텐션
* 인코더 
* 디코더

---
## 0927
* 버트(BERT)
* 한국어 임베딩

---
## 1004
* 클러스터링
* 가우시안 혼합 모델
* 베이즈 정리
* 자기 조직화 지도
* SOM 구조
* BUM(Best Matching Unit)
* 인덱싱과 슬라이싱
* 서로 다른 배열 쌓기