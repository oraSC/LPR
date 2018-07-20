import tkinter as tk
window = tk.Tk()
window.title("my_window")
window.geometry("500x500")

ca = tk.Canvas(window , bg = "yellow" , height = 200 , width = 200)
ca.pack()
#放置照片
gif_file = tk.PhotoImage(file = "E:\\ancient_docment\\win7_64\\document\\python_code\\opencv_python\\picture\\30.gif")
gif = ca.create_image(10,10,anchor = "nw",image = gif_file ,)
#画线
ca.create_line(50,50,90,90)

window.mainloop()