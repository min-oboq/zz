# def test():
#     print("A 지점 통과")
#     yield 1
#     print("B 지점 통과")
#     yield 2
#     print("C 지점 통과")
    
# output = test()
# print("D 지점 통과")
# a = next(output)
# print(a)
# print("E 지점 통과")
# b = next(output)
# print(b)
# print("F 지점 통과")
# c = next(output)

# next(output)

# user_input_a = input("정수 입력> ")

# if user_input_a.isdigit():
    
#     number_input_a = int(user_input_a)
    
#     print("원의 반지름:",number_input_a)
#     print("원의 둘레:", 2 * 3.14 * number_input_a)
#     print("원의 넓이:", 3.14 * number_input_a * number_input_a)
# else:
#     print("정수를 입력하지 않았습니다.")

# try:
#     user_input_a = int(input("정수 입력> "))
#     print("원의 반지름:",number_input_a)
#     print("원의 둘레:", 2 * 3.14 * number_input_a)
#     print("원의 넓이:", 3.14 * number_input_a * number_input_a)
# except:
#     print("무언가 잘못되었습니다.")

# list_input_a = ["52", "273", "32", "스파이", "103"]

# list_number = []
# for item in list_input_a:
    
#     try:
#         float(item)
#         list_number.append(item)
#     except:
#         pass
    
# print("{} 내부에 있는 숫자는".format(list_input_a))
# print("{}입니다".format(list_number))

# try:
#     number_input_a = int(input("정수 입력> "))
# except:
#     print("정수를 입력하지 않았습니다.")   
# else:
#     print("원의 반지름:",number_input_a)
#     print("원의 둘레:", 2 * 3.14 * number_input_a)
#     print("원의 넓이:", 3.14 * number_input_a * number_input_a)

# try:
#     number_input_a = int(input("정수 입력> "))
  
#     print("원의 반지름:",number_input_a)
#     print("원의 둘레:", 2 * 3.14 * number_input_a)
#     print("원의 넓이:", 3.14 * number_input_a * number_input_a)    
# except:
#     print("정수를 입력하지 않았습니다.") 
# else:
#     print("예외가 발생하지 않았습니다.")
# finally:
#     print("일단 프로그램이 어떻게든 끝났습니다.")

# try:
#     file = open("info.txt", "w")
   
#     file.close()
# except:
#     print("오류가 발생했습니다.")
    
# print("# 파일이 제대로 닫혔는지 확인하기")
# print("file.closed:", file.closed)

# def test():
#     print("test() 함수의 첫 줄입니다.")
#     try:
#         print("try 구문이 실행되었습니다.")
#         return
#         print("try 구문의 return 키워드 입니다.")
#     except:
#         print("except 구문이 실행되었습니다")
#     else:
#         print("else 구문이 실행되었습니다.")
#     finally:
#         print("finally 구문이 실행되었습니다.")
#     print("test() 함수의 마지막 줄입니다.")
    
# test()

# while True:
#     try:
#         print("try 구문이 실행되었습니다.")
#         break
#         print("try 구문이 break 키워드 뒤입니다.")
#     except:
#         print("except 구문이 실행되었습니다.")
#     finally:
#         print("finally 구문이 실행되었습니다.")
#     print("while 반복문의 마지막 줄입니다.")
# print("프로그램이 종료되었습니다.")

# numbers=[52, 273, 32, 103, 90, 10, 275]

# print("# (1) 요소 내부에 있는 값 찾기")
# print("- {}는 {} 위치에 있습니다.".format(52, numbers.index(52)))
# print()

# print("# (2) 요소 내부에 없는 값 찾기")
# number = 10000
# try:
#     print("- {}는 {} 위치에 있습니다.".format(number, numbers.index(number)))
# except:
#     print("- 리스트 내부에 없는 값입니다.")
# print()

# print("--- 정상적으로 종료되었습니다. ---")

# try:
#     number_input_a = int(input("정수 입력> "))
    
#     print("원의 반지름:",number_input_a)
#     print("원의 둘레:", 2 * 3.14 * number_input_a)
#     print("원의 넓이:", 3.14 * number_input_a * number_input_a)
# except Exception as exception:
    
#     print("type(exception):", type(exception))
#     print("exception:", exception)

# list_number = [52, 273, 32, 72, 100]

# try:
#     number_input = int(input("정수 입력> "))
#     print("{}번째 요소: {}".format(number_input, list_number[number_input]))
# except ValueError:
#     print("정수를 입력해주세요")
# except IndexError:
#     print("리스트의 인덱스를 벗어났어요!")

# number = input("정수 입력> ")
# number = int(number)

# if number > 0:
#     raise NotImplementedError
# else:
#     raise NotImplementedError

# from math import sin,cos,tan,floor,ceil
# print(sin(1))
# print(cos(1))
# print(tan(1))
# print(floor(2.5))
# print(ceil(2.5))

# import math as m
# print(m.sin(1))
# print(m.cos(1))
# print(m.tan(1))
# print(m.floor(2.5))
# print(m.ceil(2.5))

# import random
# print("# random 모듈")
# print("- random():", random.random())
# print("- uniform(10,20):",random.uniform(10,20))
# print("-randrange(10):",random.randrange(10))
# print("- choice([1, 2, 3, 4, 5]):", random.choice([1, 2, 3, 4, 5]))
# print("- shuffle([1, 2, 3, 4, 5]):",random.shuffle([1, 2, 3, 4, 5]))
# print("- sample([1, 2, 3, 4, 5], k=2):", random.sample([1, 2, 3 ,4, 5], k=2))

# import sys
# print(sys.argv)

# print("---")

# print("getwindowsversion:()", sys.getwindowsversion())
# print("---")
# print("copyright:", sys.copyright)
# print("---")
# print("version:", sys.version)

# sys. exit()

# import os 
# print("현재 운영체제:", os.name)
# print("현재 폴더:", os.getcwd())
# print("현재폴더 내부의 요소:", os.listdir())

# os.mkdir("hello")
# os.rmdir("hello")

# with open("original.txt", "w") as file:
#     file.write("hello")
# os.rename("original.txt", "new.txt")
# os.remove("new.txt")
# os.system("dir")

# import datetime

# print("# 현재 시작 출력하기")
# now = datetime.datetime.now()
# print(now.year, "년")
# print(now.month, "월")
# print(now.day, "일")
# print(now.hour, "시")
# print(now.minute, "분")
# print(now.second, "초")
# print()

# print("# 시간을 포맷에 맞춰 출력하기")
# output_a =now.strftime("%Y.%m.%d %H:%M:%S")
# output_b ="{}년 {}월 {}일 {}시 {}분 {}초".format(
#     now.year,
#     now.month,
#     now.day,
#     now.hour,
#     now.minute,
#     now.second)
# output_c = now.strftime("%Y{} %m{} %d{} %H{} %M{} %S{}").format(*"년월일시분초")
# print(output_a)
# print(output_b)
# print(output_c)
# print()
