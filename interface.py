import tkinter as tk
window = tk.Tk()
window.title("my_window")
window.geometry("500x500")

var = tk.IntVar()
var.set(0)
def fun():
    var.set(var.get() +1)

#创建主菜单
menubar = tk.Menu(window)
#创建子菜单
filemenu = tk.Menu(menubar ,tearoff = 0)
#添加子菜单头
menubar.add_cascade(label = "File",menu = filemenu )
#添加子菜单列表(功能)
filemenu.add_command(label = "new project" , command = fun)
filemenu.add_command(label = "open",command = fun)
filemenu.add_command(label = "save" , command = fun)




window.config(menu = menubar)
window.mainloop()