import tkinter as tk
window = tk.Tk()
window.title("my_window")
window.geometry("500x500")

var1 = tk.StringVar()
#var1.set("1234")    对tkinter变量进行赋值
la = tk.Label(window,bg = "red",width = 20, height = 2,textvariable = var1)
la.pack()
def show_checkbutton():
    if (var2.get() == 1)and(var3.get() == 0) :
        var1.set("ture")
    if (var2.get() == 0)and(var3.get() == 1) :
        var1.set("false")
    if ((var2.get() == 0) and (var3.get() == 0))or((var2.get() == 1) and (var3.get() == 1)):
        var1.set("uncertain")
#创建tkinter特有变量Int
var2 = tk.IntVar()
var3 = tk.IntVar()
c1 = tk.Checkbutton(window , text = "true" , variable = var2 , onvalue = 1 ,offvalue = 0 ,command = show_checkbutton)
c2 = tk.Checkbutton(window , text = "false" , variable = var3 , onvalue = 1 ,offvalue = 0 ,command = show_checkbutton)
c1.pack()
c2.pack()

window.mainloop()