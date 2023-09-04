# marks = [90,25,67,45,80]

# number = 0

# for mark in marks:
#     number = number +1
#     if  mark <  60:
#         continue
#         print(f"{number}학생은 합격힙니다.")
#     else:
#         print(f"{number}번 학생은 불합격 입니다.")

# for i in range(2,10):
#     for j in range(1,10):
#         print(f"{i} *{j} {i * j}", end='')
#     print('')
    
# a = [1,2,3,4]
# resuit=[]
# for num in a:
#     resuit.append(num*3)
# print(resuit)

# i =0
# while True:
#     i += 1
#     if i >= 5: break
#     print("*" * i)

# A = [70,60,55,75,95,90,80,80,85,100]
# total = 0
# for score in A:
#     total += score
    
# average = total / len(A)
# print(average)

# def sum_mul(choice, **args):
#     if choice == "sum":
#         result = 0
#         for i in args:
#             result = result + i
#     elif choice == "mul":
#         result = 1
#         for i in args:
#             result = result * i
#     return result

# result = sum_mul('sum',1,2,3,4,5)
# print(result)
# result = sum_mul('mul',1,2,3,4,5)
# print(result)

import turtle as t

# t.shape('turtle')
# angle = 60
# t.bgcolor('black')
# t.color("green")
# t.speed(0)
# for x in range(200):
#     t.forward(x)
#     t.left(angle)

# t.hideturtle();

# colors = ["red", "blue", "green", "purple"]

# t.circle(50)
# t.color("red")
# t.pensize(3)
# for i in range(0,4):
#     t.circle(i*50)
#     t.color(colors[i])

# t.shape("turle")

# n = 50
# t.bgcolor("black")
# t.color("green")
# t.speed(0)
# for x in range(n):
#     t.circle(80)
#     t.left(360/n)

# def factorial(n):
     
#      if n == 0:
#          return 1
#      else:
#          return n * factorial(n -1)
     
# print("1!:", factorial(1))
# print("2!:", factorial(2))
# print("3!:", factorial(3))
# print("4!:", factorial(4))
# print("5!:", factorial(5))

# def fibonacci(n):
#     if n == 1:
#         return 1 
#     if n == 2:
        
#         return 1
#     else:
#         return fibonacci(n-1)+fibonacci(n-2) 
    
    
# print("fibonacci(1):", fibonacci(1))
# print("fibonacci(2):", fibonacci(2))
# print("fibonacci(3):", fibonacci(3))
# print("fibonacci(4):", fibonacci(4))
# print("fibonacci(5):", fibonacci(5))

# counter = 0

# def fibonacci(n):
#     counter += 1
    
#     if n == 1:
#         return 1
#     else:
#         return fibonacci(n -1) + fibonacci(n-2)
    
# print(fibonacci(10))

# dictionary = {
#     1: 1,
#     2: 1
# }

# def fibonacci(n):
#     if n in dictionary:
#         return dictionary[n]
#     else:
#         output = fibonacci(n-1) + fibonacci(n-2)
#         dictionary[n] = output
#         return output
    
# print("fibonacci(10):", fibonacci(10))
# print("fibonacci(20):", fibonacci(20))
# print("fibonacci(30):", fibonacci(30))
# print("fibonacci(40):", fibonacci(40))
# print("fibonacci(50):", fibonacci(50))

# def flatten(data):
#     result = []
#     if type(data) is list:
#         for el in data:
#             result += flatten(el)
#     else:
#         result += [data]
#     return result

# example = ["[1,2,3],[4,[5,6]]7,[8,9]"]
# print(example)
# print(flatten(example))

# a, b = 10, 20

# print("# 교환 전 값")
# print("a:", a)
# print("b:", b)
# print()

# a, b = b, a

# print("# 교환 후 값")
# print("a:", a)
# print("b:", b)
# print()

# a, b = 97, 40
# print(divmod(a,b))
# x, y = divmod(a,b)

# def call_10_times(func):
#     for i in range(10):
#         func()
        
# def print_hello():
#     print("안녕하세요")
    
# call_10_times(print_hello)

# def power(item):
#     return item * item
# def nuder_3(item):
#     return item < 3

# power = lambda x: x * x
# unber_3 = lambda x: x < 3

# list_input_a = [1, 2, 3, 4, 5]

# output_a = map(power, list_input_a)
# print("# map() 함수의 실행 결과")
# print("map (power, list_input_a):",output_a)
# print("map(power, list_input_a):", list(output_a))
# print()

# output_b = filter(unber_3, list_input_a)
# print(output_b)
# print(list(output_b))

