import tkinter as tk
window = tk.Tk()
window.title("my_window")
window.geometry("500x500")

var1 = tk.StringVar()
#var1.set("1234")    对tkinter变量进行赋值
la = tk.Label(window,bg = "red",width = 20, height = 2,textvariable = var1)
la.pack()
def show_scale_selection(num):
    var1.set(num)

sc = tk.Scale(window , label = "my scale",orient = tk.HORIZONTAL ,  length = 200, showvalue = 0 ,from_ = 0 ,to = 10 ,
              tickinterval = 2,resolution = 0.01,command = show_scale_selection)

sc.pack()


window.mainloop()