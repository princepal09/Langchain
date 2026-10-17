from pydantic import BaseModel, Field


class Student(BaseModel):

    name : str = 'Prince'
    age: int | None = None
    email : str = "prince@gmail.com"
    cgpa : float = Field(gt = 0, lt = 10, description="A decimal value representing the cgpa of the student")

new_student = {"name" : "Prince Pal", "age" : 21, "cgpa" : 9}

student = Student(**new_student)

student_dict = dict(student)


student_dict_json = student.model_dump_json()

print(student_dict_json)