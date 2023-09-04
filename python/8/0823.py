# # # 딕셔너리를 선업합니다.
# dictionary = {
#     "name": "7D 건조 망고",
#     "type": " 당절임",
#     "ingredient": ["망고", "설탕", "메타중아황산나트륨", "치자황색소"],
#     "origin": "필리핀"
# }

# # # 출력합니다
# print("name:", dictionary["name"])
# print("type:", dictionary["type"])
# print("ingredient:", dictionary["ingredient"])
# print("origin:",dictionary["origin"])
# print()

# # 값을 변경합니다.
# dictionary["name"] = "8D 건조 망고"
# print("name:", dictionary["name"])

# import random
# # 임의의 실수를 반환한다.
# print(random.random())

# # a~b사이의 숫자를 반환한다.
# print(random.randint(0, 2))
# print(random.randrange(0, 3))

# #로또 번호 뽑기 6개, 청소 당번 번호 뽑기
# numbers = []
# while len(numbers) < 7:
#     number = ranbom.randint(1, 10)
#     if number not in numbers:
#         numbers.append(numbers)
        
# print(numbers)

# dictionary = {
#     "name": "7D 건조 망고",
#     "type": "당절임",
#     "ingredient": ["망고", "설탕", "메타중아황산나트륨", "치자황색소"],
#     "origin": "필리핀"
# }

# key = input("> 접근하고자 하는 키:")

# if key in dictionary:
#     print(dictionary[key])
# else:
#     print("존재하지 않는 키에 접근하고 있습니다.")

# dictionary = {
#     "name": "7D 건조 망고",
#     "type": "당절임",
#     "ingredient": ["망고", "설탕", "메타중아황산나트륨", "치자황색소"],
#     "origin": "필리핀"
# }

# value = dictionary.get("존재하지 않는 키")
# print("값:",value)

# if value == None:
#     print("존재하지 않는 키에 접근했었습니다.")

# dictionary = {
#     "name": "7D 건조 망고",
#     "type": "당절임",
#     "ingredient": ["망고", "설탕", "메타중아황산나트륨", "치자황색소"],
#     "origin": "필리핀"
# }

# for key in dictionary:
#     print(key, ":", dictionary[key])

# array = [273, 32, 103, 57, 52]

# for i in range(len(array)):
#     print("{}번째 반복: {}".format(i, array[i]))

# for i in range (4, 0 -1, -1):
#     print("현재 반복 변수: {}".format(i))

# numbers = [5, 15, 6, 20, 7, 25]

# for number in numbers:
#     if number < 10:
#         continue
#     print(number)

# numbers = [103, 52, 273, 32, 77]

# print(min(numbers))
# print(max(numbers))
# print(sum(numbers))

# example_list = ["요소A", "요소B", "요소C"]

# print("#단순 출력")
# print(example_list)
# print()

# print("#enumerate() 함수 적용 출력")
# print(enumerate(example_list))
# print()

# print("#list() 함수로 강제 변환 출력")
# print(list(enumerate(example_list)))
# print()

# print("# 반복문과 조합하기")
# for i, value in enumerate(example_list):
#     print("{}번째 요소는 {}입니다.".format(i, value))

# example_dictionary = {
#     "키A" : "값A",
#     "키B" : "값B",
#     "키C" : "값C",
# }

# print("#딕셔너리의 items() 함수")
# print("items():", example_dictionary.items())
# print()

# print("# 딕셔너리의 items() 함수와 반복문 조합하기")

# for key, element in example_dictionary.items():
#     print("dictionary[{}]={}".format(key, element))
    
# array = []

# for i in range(0, 20, 2):
#     array.append(i*i)
    
# print(array)

# array = [i * i for i in range(0, 20, 2)]
# print(array)

# array = ["사과", "자두", "초콜릿", "바나나", "체리"]
# ouput = []
# for fruit in array:
#     if fruit != "초콜릿":
#         ouput.append(fruit)
        
# print(ouput)

# # output = [fruit for fruit in array if fruit != "초콜릿"]

# # print(output)

# numbers = [1, 2, 3, 4, 5, 6]
# r_num = reversed(numbers)

# print("reversed_numbers :", r_num)
# print(next(r_num))
# print(next(r_num))
# print(next(r_num))
# print(next(r_num))
# print(next(r_num))


# a_input=input("1숫자 입력")
# b_input=input("2숫자 입력")

# num_a = int(a_input)
# num_b = int(b_input)

# print(num_a + num_b )

# print(num_a * num_b)

# print(num_a / num_b)

# print(num_a % num_b)

# x = 3
# y = 2
# print(x>y)
# print(x<y)

# if x > y:
#     print("참")
# else:
#     ("거짓")
    
# a_input=input("1숫자 입력")
# b_input=input("2숫자 입력")

# num_a = int (a_input)
# num_b = int (b_input)

# if num_a > 0:
#     print("참")
# else:
#     print("error")

# money = input("입력")

# money = int (money)

# if money >= 3000:
#     print("택시 타")
# else:
#     print("걸어가")

# money = 2000
# card = 1

# if money >= 3000 or card:
#     print("택시 타")
# else:
#     print("걸어 가")
    
# pocket = ['paper', 'cellphone', 'money']
# dle pocket[2]
# print(pocket)
# if 'money' in pocket:
#     print("택시 타")
# else:
#     print("걸어가")
    
