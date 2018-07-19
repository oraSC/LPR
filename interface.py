import tkinter as tk
window = tk.Tk()
window.title("my_window")
window.geometry("500x500")

var1 = tk.StringVar()
#var1.set("1234")    对tkinter变量进行赋值
la = tk.Label(window,bg = "red",width = 20, height = 2,textvariable = var1)
la.pack()


#可以添加回调函数
r1 = tk.Radiobutton(window, text = "true",variable = var1 , value = "true")
r2 = tk.Radiobutton(window, text = "true",variable = var1 , value = "false")

r2.pack()
r1.pack()

window.mainloop()