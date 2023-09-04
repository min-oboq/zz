# from urllib import request
# from bs4 import BeautifulSoup
# target = request.urlopen("https://discord.com/channels/1141531162607624234/1141531163165475040/1145879415432163339")
# soup = BeautifulSoup(target, "html.Parser")

# for location in soup.select("location"):
#     print("도시:", location.select_one("city").string)
#     print("날씨:", location.select_one("wf").string)
#     print("최저기온:", location.select_one("tmn").string)
#     print("최고기온:", location.select_one("tmx").string)
#     print()
    
# from flask import Flask
# from urllib import request
# from bs4 import BeautifulSoup

# app = Flask(__name__)
# @app.route("/")

# def hello():
#     target = request.urlopen("https://discord.com/channels/1141531162607624234/1141531163165475040/1145879415432163339")
#     soup = BeautifulSoup(target, "html.parser")
    
#     output = ""
#     for location in soup.select("location"):
#         output += "<h3>{}</h3>".format(location.select_one("city").string)
#         output += "날씨: {}<br/>".format(location.select_one("wf").string)
#         output += "최저/최고 기온: {}/{}"\
#             .format(\
#                 location.select_one("tmn").string,
#                 location.select_one("tmx").string
#                 )
#         output += "<hr/>"
#     return output

# PI = 3.141592 

# def number_input():
#     output = input("숫자 입력> ")
#     return float(output)

# def get_circumference(radius):
#     return 2 * PI * radius

# def get_circle_area(radius):
#     return PI * radius * radius

# if __name__ == "__main__":
#     print("get_circumference(10):", get_circumference(10))
#     print("get_circle_area(10):", get_circle_area(10))

# import test_module

# print("# 메인의 __name__ 출력하기")
# print(__name__)
# print()

# ψ(._. )> ==3

# students = [
#     {"name": "윤인성", "Korean":87, "math":98, "english":88, "science":95},
#     {"name": "연하진", "Korean":92, "math":98, "english":96, "science":98},
#     {"name": "구지연", "Korean":76, "math":96, "english":94, "science":90},
#     {"name": "나선주", "Korean":98, "math":92, "english":96, "science":92},
#     {"name": "윤아린", "Korean":95, "math":98, "english":98, "science":98},
#     {"name": "윤명월", "Korean":64, "math":88, "english":92, "science":92},
# ]

# print("이름", "총점", "평균", sep="\t")

# for student in students:
#     score_sum = student["Korean"]+student["math"]+\
#         student["english"]+student["science"]
#     score_average = score_sum / 4
    
#     print(student["name"], score_sum, score_average, sep="\t")

# class Student:
#     def __init__(self, name, korean, math, english, science):
#         self.name = name
#         self.korean = korean
#         self.math = math
#         self.english = english
#         self.science = science
#     def get_sum(self):
#         return self.korean + self.math +\
#             self.english + self.science
#     def get_average(self):
#         return self.get_sum() / 4
#     def to_string(self):
#         return "{}\t{}\t{}".format(
#             self.name,
#             self.get_sum(),
#             self.get_average())  
              
# students = [
#     Student("윤인성", 87, 98, 88, 95),
#     Student("연하진", 92, 98, 96, 98),
#     Student("구지연", 76, 96, 94, 90),
#     Student("나선주", 98, 92, 96, 92),
#     Student("윤아린", 95, 98, 98, 98),
#     Student("윤명월", 64, 88, 92, 92)
# ]
 
# print("이름", "총점", "평균", sep="\t")
# for student in students:
     
#      print(student.to_string())
        
# students[0].name
# students[0].korean
# students[0].math
# students[0].english
# students[0].science

# class Human:
#     def __init__(self):
#         pass
# class Student(Human):
#     def __init__(self):
#         pass
    
# student = Student()

# print("isinstance(student, Human):", isinstance(student, Human))
# print("type(student) == Human:", type(student) == Human)

# class Student:
#     def study(self):
#         print("공부를 합니다")
        
# class Teacher:
#     def teach(self):
#         print("학생을 가르칩니다")
        
# classroom = [Student(), Student(), Teacher(), Student(), Student(),]

# for person in classroom:
#     if isinstance(person, Student):
#         person.study()
#     elif isinstance(person, Teacher):
#         person.teach()

# class Student:
#     def __init__(self, name, korean, math, english, science):
#         self.name = name
#         self.korean = korean
#         self.math = math
#         self.english = english
#         self.science = science
        
#     def get_sum(self):
#         return self.korean + self.math +\
#             self.english + self.science
            
#     def get_average(self):
#         return self.get_sum() / 4
    
#     def to_string(self):
#         return "{}\t{}\t{}".format(
#             self.name,
#             self.get_sum(),
#             self.get_average()) 
    
#     def __eq__(self, value):
#         return self.get_sum() == value.get_sum()
#     def __ne__(self, value):
#         return self.get_sum() != value.get_sum()
#     def __gt__(self, value):
#         return self.get_sum() > value.get_sum()
#     def __ge__(self, value):
#         return self.get_sum() >= value.get_sum()
#     def __lt__(self, value):
#         return self.get_sum() < value.get_sum()
#     def __le__(self, value):
#         return self.get_sum() <= value.get_sum()
        
# students = [
#     Student("윤인성", 87, 98, 88, 95),
#     Student("연하진", 92, 98, 96, 98),
#     Student("구지연", 76, 96, 94, 90),
#     Student("나선주", 98, 92, 96, 92),
#     Student("윤아린", 95, 98, 98, 98),
#     Student("윤명월", 64, 88, 92, 92)
# ]

# student_a = Student("윤인성", 87, 98, 88, 95),
# student_b = Student("연하진", 92, 98, 96, 98),

# print("student_a == student_b = ", student_a == student_b)
# print("student_a != student_b = ", student_a != student_b)
# print("student_a > student_b = ", student_a > student_b)
# print("student_a >= student_b = ", student_a >= student_b)
# print("student_a < student_b = ", student_a < student_b)
# print("student_a <= student_b = ", student_a <= student_b)

# print("이름", "총점", "평균", sep="\t")
# for student in students:
#     print(str(student))

# class Student:
#     count = 0
#     students = []
    
#     @classmethod
#     def __init__(self, name, korean, math, english, science):
#         self.name = name
#         self.korean = korean
#         self.math = math
#         self.english = english
#         self.science = science
        
#         Student.count += 1
#         print("{}번째 학생이 생성되었습니다.".format(Student.count))
    
#     students = [
#     Student("윤인성", 87, 98, 88, 95),
#         Student("연하진", 92, 98, 96, 98),
#         Student("구지연", 76, 96, 94, 90),
#         Student("나선주", 98, 92, 96, 92),
#         Student("윤아린", 95, 98, 98, 98),
#         Student("윤명월", 64, 88, 92, 92)
# ]
    
# print()
# print("현재 생성된 총 학생 수는 {}명 입니다.".format(Student.count))