import tkinter as tk
window = tk.Tk()
window.title("my_window")
window.geometry("500x500")

var1 = tk.StringVar()
#var1.set("1234")    对tkinter变量进行赋值
la = tk.Label(window,bg = "red",width = 20, height = 2,textvariable = var1)

def show_selection():
    var = lb.get(lb.curselection())
    var1.set(var)

la.pack()
bu = tk.Button(window,bg = "yellow",width = 20,height = 2 ,text = "selection",command = show_selection)
bu.pack()
var2 = tk.StringVar(0)
var2.set((11,22,33,44))

lb = tk.Listbox(window,listvariable = var2,bg = "blue")
#插入，后面的内容往后退
lb.insert(1,"hello")

lb.pack()

window.mainloop()