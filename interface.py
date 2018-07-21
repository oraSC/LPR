import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image,ImageTk

#定义一个Interface类，继承ttk.Frame
class Interface(ttk.Frame):
        pic_path = ""
        viewhigh = 600
        viewwide = 600
        update_time = 0
        thread = None
        thread_run = False
        camera = None
        color_transform = {"green": ("绿牌", "#55FF55"), "yello": ("黄牌", "#FFFF00"), "blue": ("蓝牌", "#6666FF")}
        #Interface 构造函数
        def __init__(self, win):
            #初始化Interface ttk.Frame部分，将Interface置于实例中master之中
            ttk.Frame.__init__(self, win)
            #创建Frame子部件，并将各个子Frme部件置于实例类中
            frame_left = ttk.Frame(self)
            frame_right1 = ttk.Frame(self)
            frame_right2 = ttk.Frame(self)
            #窗口最大化
            #win.state("zoomed")
            #放置Interface
            #padx,pady为  单元行间距
            self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
            frame_left.pack(side=LEFT, expand=1, fill=BOTH)
            frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
            frame_right2.pack(side=RIGHT, expand=0)
            #frame_left布局
            ttk.Label(frame_left, text='原图：').pack(anchor="nw")
            self.image_ctl = ttk.Label(frame_left)
            self.image_ctl.pack(anchor="nw")

            #frame_right1布局
            #grid布局，指定行指定列，sticky  对齐方式
            ttk.Label(frame_right1, text='车牌位置：').grid(column=0, row=0, sticky=tk.W)
            self.roi_ctl = ttk.Label(frame_right1)
            self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
            ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
            self.r_ctl = ttk.Label(frame_right1, text="")
            self.r_ctl.grid(column=0, row=3, sticky=tk.W)
            self.color_ctl = ttk.Label(frame_right1, text="", width="20")
            self.color_ctl.grid(column=0, row=4, sticky=tk.W)
            #frame_right2
            from_pic_ctl = ttk.Button(frame_right2, text="来自图片", width=20, command=self.from_pic)
            # from_vedio_ctl = ttk.Button(frame_right2, text="来自摄像头", width=20, command=self.from_vedio)
            from_pic_ctl.pack(anchor="se", pady="1")

        def get_imgtk(self, img_bgr):
            #bgr颜色空间转化为rgb颜色空间
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            #img 的变量类型为numpy.ndarray
            #im 的变量类型为PIL.Image.Image
            #image 的变量类型为PIL.ImageTk.PhotoImage
            im = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage( image = im)
            wide = imgtk.width()
            high = imgtk.height()
            if wide > self.viewwide or high > self.viewhigh:
                wide_factor = self.viewwide / wide
                high_factor = self.viewhigh / high
                factor = min(wide_factor, high_factor)
                wide = int(wide * factor)
                if wide <= 0: wide = 1
                high = int(high * factor)
                if high <= 0: high = 1
                #改变尺寸大小
                im = im.resize((wide, high), Image.ANTIALIAS)
                imgtk = ImageTk.PhotoImage(image=im)
            return imgtk

        def from_pic(self):
            self.thread_run = False
            self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
            if self.pic_path :
                #np.fromfile(self.pic_path, dtype=np.uint8)读取的是一个numpy.ndarray , 但未进行编码处理
                #需要结合cv2.imdecode()进行对数据数据的编码读取
                #img_bgr = cv2.imdecode(np.fromfile(self.pic_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                img_bgr = cv2.imread(self.pic_path,cv2.IMREAD_COLOR )
                #将cv2 format（numpy.ndarray） 格式转化为tkinter image
                self.imgtk = self.get_imgtk(img_bgr)
                self.image_ctl.configure(image=self.imgtk)


if __name__ == '__main__':
    win = tk.Tk()
    win.title("车牌识别")
    win.geometry("500x500")
    interface = Interface(win)

    win.mainloop()
