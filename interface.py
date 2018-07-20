import tkinter as tk
window = tk.Tk()
window.title("my_window")
window.geometry("500x500")

var = tk.IntVar()
var.set(0)
def fun():
    var.set(var.get() +1)

class employee:
    #所有employee公有元素
    employee_num = 0
    #类的构造函数，self代表假定的实类
    def __init__(self,name,salary):
        self.name = name
        self.salary = salary
        employee.employee_num += 1
        self.NO = employee.employee_num
    #类的析构函数
    # def __del__(self):
    #     print(self.name,"have been fired")
    #     employee.employee_num -= 1
    def displayemployee(self):
        print("name:",self.name,"salary:",self.salary,"NO.",self.NO)

em1 = employee("陈志伟","1k")
em2 = employee("路痴不橙","2k")

em1.displayemployee()
em2.displayemployee()
del em2