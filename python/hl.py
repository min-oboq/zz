# str_input=input("원의 반지름 입력> ")
# num_input=float(str_input)
# print()
# print("반지름:", num_input)
# print("둘레:",2*3.14*num_input)
# print("넓이:",3.14*num_input**2)

# a=input("문자열 입력>")
# b=input("문자열 입력>")

# print(a,b)
# c=a
# a=b
# b=c
# print(a,b)

# formt() 함수로 숫자를 문자열로 변환하기
# string_a="{}".format(10)

# #출력하기
# print(string_a)
# print(type(string_a))

#format() 함수로 숫자응 문자열로 변환하기
# format_a="{}만 원".format(5000)
# format_b="파이썬 열공하여 첫 연봉 {}만 원 만들기".format(5000)
# format_c="{}{}{}".format(3000, 4000, 5000)
# format_d="{}{}{}".format(1,"문자열",True)

# #출력하기
# print(format_a)
# print(format_b)
# print(format_c)
# print(format_d)

#조합하기
# output_h="{:+5d}".format(52)
# output_i="{:+5d}".format(-52)
# output_j="{:=+5d}".format(52)
# output_k="{:=+5d}".format(-52)
# output_l="{:+05d}".format(52)
# output_m="{:+05d}".format(-52)

# print("# 조합하기")
# print(output_h)
# print(output_i)
# print(output_j)
# print(output_k)
# print(output_l)
# print(output_m)

# output_a="{:15.3f}".format(52.273)
# output_b="{:15.2f}".format(52.273)
# output_c="{:15.1f}".format(52.273)

# print(output_a)
# print(output_b)
# print(output_c)

# a="Hello Python Programming...!"
# b=a.upper()
# print(a)
# print(b)

# c=a.lower()
# print(c)

# input_a= """"
#     안녕하세요
# 문자열의 함수를 알아봅니다
# """

# print(input_a)
# print(input_a.strip())

# output_a="안녕안녕하세요".find("안녕")
# print(output_a)

# output_b="안녕안녕하세요".rfind("안녕")
# print(output_b)

# print("안녕"in"안녕하세요")
# print("잘자"in"안녕하세요")

# a="10 20 30 40 50".split(" ")
# print(a)
# b="10,20,30,40,50".split(" , ")
# print(b)

# string="hello"

# #string.upper()를 실행하고, string 출력하기
# string.upper()
# print("A지점:",string)

# #string.upper() 실행하기
# print("B지점:",string.upper())

# a=input(">1번째 숫자: ")
# b=input(">2번째 숫자: ")
# print()

# print("{}+{}+{}".format(a,b,int(a)+int(b)))

# print(10==100)
# print(10!=100)
# print(10<100)
# print(10>100)
# print(10<=100)
# print(10<=100)

# print("가방"=="가방")
# print("가방"!="하마")
# print("가방"<"하마")
# print("가방">"하마")

# if True:
#     print("True입니다...!")
#     print("정말 True입니다...!")
    
# if False:
#     print("False입니다...!")
    

# number=input("정수 입력>")
# number=int(number)

# if number>0:
#     print("양수입니다")
    
# if number <0:
#     print("음수입니다")

# if number ==0:
#     print("0입니다")

# number=input("정수 입력>")
# number=int(number)

# if number > 0 and number % 5 == 0:
#     print("5의 배수 입니다")
    
# import datetime

# now=datetime.datetime.now()
# print(now.year)
# print(now.month)
# print(now.day)
# print(now.hour)
# print(now.minute)
# print(now.second)

# import datetime
# now=datetime.datetime.now()

# print("{}년 {}월 {}일 {}시 {}분 {}초".format(
#     now.year,
#     now.month,
#     now.day,
#     now.hour,
#     now.minute,
#     now.second
# ))

# if now.hour < 12:
#     print("현재 시각은 {}시로 오전입니다!".format(now.hour))
    
# if now.hour >= 12:
#     print("현재 시각은 {}시로 오후입니다!".format(now.hour))

# str_age=input("고궁의 입장하는 분의 나이를 입력해주세요>")
# num_age=int(str_age)

# price=5000

# if num_age>=65 or num_age<=5:
#     print=0
    
# print("고궁의 입장료는 {}원 입니다".format(price))

# a=int(input("> 1번쨰 숫자:"))
# b=int(input("> 2번째 숫자:"))

# if a>b:
#     print(f"처음 입력했던 {a}가 {b}보다 더 크다")
    
# if a<b:
#     print(f"두 번째로 입력했던 {b}가 {a}보다 더 크다")

# number=input("정수 입력>")
# number=int(number)

# if number % 2 == 0:
#     print(f"{number}는 짝수입니다.")
    
# else:
#     print(f"{number}는 홀수입니다.")
    
# money=input("돈을 입력")
# money=int(money)

# if money >= 5000:
#     print("택시를 탄다")
# elif money >=2000:
#     print("버스를 탄다")
# else:
#     print("걸어간다")

# str_score=input("성적을 입력해주세요")
# score=int(str_score)

# if score>=90:
#     print(f"아이패드를 받았다!")
# elif score>=80:
#     print(f"애플워치를 받았다!")
# elif score>=70:
#     print(f"아무일도 일어나지 않는다")
# elif score>=60:
#     print(f"용돈 차감!")
# else:
#     print(f"외출금지!!")

# score=float(input("학점 입력> "))

# if score ==4.5:
#     print("신")
# elif 4.2<= score:
#     print("교수님의 사랑")
# elif 3.5 <= score:
#     print("현 체제의 수호자")
# elif 2.8 <= score:
#     print("일반인")
# elif 2.3 <= score:
#     print("일탈을 꿈구는 소시민")
# elif 1.75 <= score:
#     print("오락문화의 선구자")
# elif 1.0 <= score:
#     print("불가촉천민")
# elif 0.5 <= score:
#     print("자벌레")
# elif 0 < score:
#     print("플랑크톤")
# else:
#     print("시개를 앞서가는 혁명의 씨앗")
    

# number_first = float(input("1숫자 입력> "))
# number_second = float(input("2숫자 입력> "))
# number_third = float(input("3숫자 입력> "))

# if number_first > number_second:
#     if number_first > number_first:
#         print(f"가장 큰 숫자는 {number_first}")
#     else:
#         print(f"가장 큰 숫자는 {number_third}")
# else:
#     if number_second > number_third:
#         print(f"가장 큰 숫자는 {number_second}")
#     else:
#         print(f"가장 큰 숫자는 {number_third}")

# max_number = int(input("1번 숫자를 입력하세요> "))

# a= int (input("2번 숫자를 입력하세요"))

# if a > max_number:
#     max_number = a
    
# a = int (input("3번 숫자를 입력하세요"))

# if a > max_number:
#     max_number = a
# print(f"가장 큰 숫자는 {max_number}")

# coke = 1000
# cidar = 800
# fanta = 900
# milkiss = 700
# soleyes = 600
# samdasoo = 500

# print("---"*10)
# print("음료수 자판기 입니다.")
# print("---"*10)
# money = int(input("돈을 입력해주세요>"))
# print("---*10")
# if money < 1000:
#     print("돈이 부족합니다. 1000원 이상으로 입력해주세요")
# else:
#     print("---"*10)
#     print(f"""
#      | 1번 코카콜라 {coke}원 |   
#      | 2번 사이다 {cidar}원 |  
#      | 3번 환타 {fanta}원 |       
#      | 4번 밀키스 {milkiss}원 |       
#      | 5번 솔의눈 {soleyes}원 |       
#      | 6번 삼다수 {samdasoo}원 |       
#          """)
#     print("---"*10)
#     choice = int(input("음료를 선택 해주세요"))
    
# if choice == 1:
#     money = money - coke
#     print("코카콜라를 선택하셨습니다.")
# elif choice ==2:
#     money = money - cidar
#     print("사이다를 선택하셨습니다")
# elif choice ==3:
#     money = money - fanta
#     print("환타를 선택하셨습니다")
# elif choice ==4:
#     money = money - milkiss
#     print("밀키스를 선택하셨습니다")
# elif choice ==5:
#     money = money - soleyes
#     print("솔의눈을 선택하셨습니다")
# elif choice ==6:
#     money = money -samdasoo
#     print("삼다수를 선택하셨습니다")
# else:
#     print("번호를 잘못 선택하셨습니다")
    
# print(f"잔돈 {money} 원")

# number = input("정수 입력>")
# number = int(number)

# #조건문 사용
# if number > 0:
#     print("")
# else:
#     pass

# list_a = [273, 32, 103, "문자열", True, False]
# print(list_a[a])
# print(list_a[1])
# print(list_a[2])
# print(list_a[1:3])

# list_a = [[1,2,3], [4,5,6], [7,8,9]]
# print(list_a[1])
# print(list_a[1][2])
# print(list_a[2][1])

# list_a=[1,2,3]
# list_b=[4,5,6]

# print("#리스트")
# print("list_a=",list_a)
# print("list_b=",list_b)
# print()

# # 기본 연산자
# print("#리스트 기본 연산자")
# print("list_a+list_b=",list_a+list_b)
# print("list_a*3",list_a*3)
# print()

# #함수
# print("#길이 구하기")
# print("len(list_a)=",len(list_a))

# list_a = [1,2,3]

# print("#리스트 뒤에 요소 추기하기")
# list_a.append(4)
# list_a.append(5)
# print(list_a)
# print()

# print("#리스트 중간에 요소 추가하기")
# list_a.insert(0,10)
# print(list_a)

# treeHit = 0
# while treeHit <10:
#     treeHit = treeHit+1
#     print("나무를 %d번 찍었습니다."%treeHit)
#     if treeHit ==10:
#         print("나무 넘어갑니다.")

# prompt = """
# ...1.Add
# ...2.Del
# ...3.List
# ...4.Quit
# ...
# ...Enter number"""

# number = 0
# while number !=4:
#     print(prompt)
#     number=int(input())
    
# coffee = 10
# money = 300
# while money:
#     print("돈을 받았으니 커피를 줍니다.")
#     coffee = coffee -1
#     print(f"남은 커피의 양은 {coffee}개입니다.")
#     if not coffee:
#         print("커피가 다 떨어졌습니다. 판매를 중지합니다.")
#         break

# coffee = 10
# while True:
#     money = int(input("돈을 넣어 주세요:"))
#     if money==300:
#         print("커피를 줍니다.")
#         coffee = coffee-1
#     elif money >300:
#         print(f"거스름돈 {money-300}를 주고 커피를 줍니다.")
#         coffee=coffee-1
#     else:
#         print(f"돈{money}을 다시 돌려주고 커피를 주지않습니다")
#     if not coffee:
#         print("커피가 다 떨어졌습니다. 판매를 중지합니다.")
#     break

# print(f"남은 커피의 양은{coffee}개 입니다.")

# i = 0
# while i < 10:
#     print(f"2*{i}={2*i}")
#     #print("2*{}={}".format(i,2*i))
#     i+=1
    
# dan = int(input("몇 단을 출력할까요? >"))
# i=1
# while i < 10:
#     print(f"{dan}*{i}={dan*i}")
#     # print("2*{}={}".format(i,2*i))
#     i+=1

# #시간과 관련된 기능을 가져옵니다.
# import time

# #변수를 선언합니다.
# number = 0

# #5초 동안 반복합니다.
# target_tick = time.time()+5
# while time.time() < target_tick:
#     number += 1

# print(f"5초 동안 {}번 반복했습니다".format(number))

# i =dan = int(input("몇 단을 출력할까요? >"))
# i=1
# while i < 10:
#     print(f"{dan}*{i}={dan*i}")
#     # print("2*{}={}".format(i,2*i))
#     i+=1 0

# while True:
#     print("{}번째 반복문입니다.".format(i))
#     i = i + 1
#     input_text = input("> 종료하시겠습니까?(y/n):")
#     if input_text in ["y", "Y"]:
#         print("반복을 종료합니다.")
#         break

# i = 0
# while i < 10:
#     print("*"*i,end='')
#     print()
#     i += 1
    
# i = 10
# while i > 0:
#     print("*"*i,end='')
#     print()
#     i -= 1
 
# j = 0
# i = 0
# while i < 10:
#     j = 0
#     print(f"i={i}")
#     while j <10:
#          print(f"j={j}\t",end='')
#          j += 1
#     print()
    # i += 1

# dan = 2
# i = 1
# while dan < 10:
#     i = 1
#     while i < 10:
#         print(f"{dan} * {i} = {dan * i} ",end='')
#         i += 1
#     print()
#     dan += 1