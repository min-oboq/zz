# def print_3_times():
#     print("안녕하세요")
#     print("안녕하세요")
#     print("안녕하세요")
    
# print_3_times()

# def printn_n_times(value, n):
#     for i in range(n):
#         print(value)
        
# printn_n_times("안녕하세요", 5)


# def print_n_times(n, *values):
    
#     for i in range(n):
        
#         for value in values:
#             print(value)
            
#         print()
        
# print_n_times(3, "안녕하세요", "즐거운", "파이썬 프로그래밍")

# def print_n_times(value, n=2):
    
#     for i in range(n):
#         print(value)
        
# print_n_times("안녕하세요")
    
# def print_n_times(*values, n=2):
    
#     for i in range(n):
        
#         for value in values:
#             print(value)
            
#             print()
            
# print_n_times("안녕하세요", "즐거운", "파이썬 프로그래밍", n=3)

# def return_test():
#     return 100

# value = return_test()
# print(value)

# def sum_all(start, end):
#     output = 0
#     for i in range(start, end + 1):
#         output += i
        
#     return output

# print("0 to 100:", sum_all(0,100))
# print("0 to 1000:", sum_all(0,1000))
# print("50 to 100:", sum_all(50,100))
# print("50 to 1000:", sum_all(50,1000))

# file = open ("basic.txt", "w")

# file.write("Hello Python Programming...!")

# file.close

# with open("basic.txt", "r") as file:
#     contents = file.read()
# print(contents)

# import random

# hanguls = list("가나다라마바사아자차카타파하")

# with open("info.txt", "w") as file:
#     for i in range(1000):
        
#         name = random.choice(hanguls) + random.choice(hanguls)
#         weight = random.randrange(40, 100)
#         height = random.randrange(140, 200)
        
#         file.write("{}, {}, {}\n".format(name, weight, height))
        
# with open("info.txt", "r") as file:
#     for line in file:
        
#         (name, weight, height) = line.strip().split(", ")
        
#         if (not name) or (not weight) or (not height):
#             continue
        
#         bmi = int(weight) / ((int(height) / 100) **2)
#         result = ""
#         if 25<= bim:
#             result = "과체중"
#         elif 18.5 <= bmi:
#             result = "정상체중"
#         else:
#             result = "저체중"
#         print('\n'.join([
#             "이름: {}"
#             "몸무게: {}"
#             "키: {}"
#             "BMI: {}"
#             "결과: {}"
#         ]).format(name, weight, height, bmi, result))
#         print()  
        
# treeHit  = 0
# while treeHit < 10:
#     treeHit = treeHit +1
#     print(f"나무를 {treeHit} 번 찍었다.")
#     if treeHit == 10:
#         print("나무 넘어간다")

# prompt = """
# 1.Add
# 2.Del
# 3.List
# 4.Quit

# ...Enter number:
# """

# number = 0
# while number != 4:
#     print(prompt)
#     number = int(input())


# number = 0
# while number < 10:
#     number += 1
#     print(f"{number}")
#     if number == 10:
#         print("{number}")

# number = 11
# while number > 1:
#     number -= 1
#     print(f"{number}")
#     if number == 11:
#         print("{number}")

# coffee = 10
# money = 300
# while money:
#     print("돈을 받았으니 커피를 준다")
#     coffee = coffee-1
#     print("남은 커피의 양은 %d개입니다."%coffee)
#     if not coffee:
#         print("커피가 다 떨어졌습니다. 판매를 중지합니다")
#         break

# coffee = 10

# while True:
#     money = int(input("돈을 넣어 주세요:"))
#     if money == 300:
#         print("커피를 준다")
#         coffee = coffee -1
#     elif money > 300:
#         print(f"거스름돈 {money - 300}")
#         coffee = coffee -1
#     else:
#         print(f"돈 다시 줌 {money}")
#     if not coffee:
#         print("커피 없어 판매 중지")
#         break
#     print(f"남은 커피의 양 {coffee}")

# for i in range (1, 101):
#     if i % 2 == 0:
#         print(i)

# a = 0
# for i in range(1, 10):
#     print(f"*{i}")

# print(a)

# output = ""

# for i in range(10, 0, -1):
#     for j in range(0, i):
#         output +="*"
#     output += "\n"
    
# print(output)

# output = ""

# for i in range(1, 15):
#     for j in range(14, i, -1):
#         output += ' '
#     for k in range(0, 2 * i -1):
#         output += '*'
#     output += '\n'
        
# print(output)

# for i in range(1, 15):
#     for j in range(14, i, -1):
#         output += ' '
#     for k in range(0, 2 * i -1):
#         output += '*'
#     output += '\n'
        
# print(output)