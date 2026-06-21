from pydantic import BaseModel , EmailStr , Field
from typing import Optional 

class student( BaseModel):
    name: str = 'subhan' #  default value
    age: Optional[int] = None
    email: Optional[EmailStr] = None
    gpa: float = Field(default=3.4 , gt = 0 , lt = 5 , description="GPA must be between 0 and 5")
new_student = student(name='32', email='32e@example.com' ,)

print(type(new_student))


