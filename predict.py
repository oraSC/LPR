import cv2
import json
from matplotlib import pyplot as plt
from pylab import mpl         #改matplotlib显示字体格式
import numpy as np
import math

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体（仿宋）
class carpredictor:
    #高斯模糊核
    gaussian_blur = 0


    def __init__(self):
        # 车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
        #打开配置好的文件，读取json包#文件类型不一定是.js  / .txt也行
        f = open("config.js")
        #将config.js读取的json包转化为字典类型
        j = json.load(f)
        for c in j["config"]:
            if c["open"]:
                self.cfg = c.copy()
                #读取相关参数
                #高斯核
                self.gaussian_blur = self.cfg["gaussian_blur"]
                break
        else:
            raise RuntimeError("没有设置有效配置参数")
    #图像特定颜色区域提取（蓝色与绿色）
    def color_mask(self , img):
        # 将颜色空间转换为hsv 提取：蓝色、黄色、白色、黑色、绿色
        blue_lower = np.array([100, 43, 46])
        blue_upper = np.array([124, 255, 255])
        yellow_lower = np.array([26, 43, 46])
        yellow_upper = np.array([34, 255, 255])
        white_lower = np.array([0, 0, 221])
        white_upper = np.array([180, 30, 255])
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255.46])
        green_lower = np.array([30, 43, 46])
        green_upper = np.array([77, 255, 255])


        HSV_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        blue_HSV_img = cv2.inRange(HSV_img, blue_lower, blue_upper)
        green_HSV_img = cv2.inRange(HSV_img, green_lower, green_upper)
        # white_HSV_img = cv2.inRange(HSV_img, white_lower, white_upper)
        color_HSV_img = cv2.bitwise_or(blue_HSV_img, green_HSV_img)
        # color_HSV_img = cv2.bitwise_or(color_HSV_img, white_HSV_img)
        # 膨胀，解决绿色车牌渐变绿的特征 # Y方向膨胀，解决绿色渐变色车牌顶部为白色特点
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        color_HSV_img = cv2.dilate(color_HSV_img, element, iterations=1)
        # 双边模糊，防止模糊边界
        color_HSV_img = cv2.bilateralFilter(color_HSV_img, 20, 75, 75)
        color_mask_img = cv2.bitwise_and(img.copy(), img.copy(), mask=color_HSV_img)
        return color_mask_img

    def find_carplate(self , img):

        origin_img = img.copy()
        #高斯模糊处理
        gauss_img = cv2.GaussianBlur(origin_img.copy(), (self.gaussian_blur, self.gaussian_blur), 0)
        # 图像特定颜色区域提取（蓝色与绿色）
        color_mask_img = self.color_mask(gauss_img)
        #中值滤波
        media_blue_img = cv2.medianBlur(color_mask_img,3)
        cv2.imshow("HSV_img",media_blue_img)
        #高斯模糊处理
        gauss_img = cv2.GaussianBlur(media_blue_img, (self.gaussian_blur,self.gaussian_blur) , 0)
        gray_gauss_img = cv2.cvtColor(gauss_img.copy(),cv2.COLOR_BGR2GRAY)
        # 边缘检测
        sobel_img = cv2.Sobel(gray_gauss_img, cv2.CV_8U, 1, 0, ksize=3)
        # 二值化
        ret, thresh_img = cv2.threshold(sobel_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #形态学变换

        # morphology_img1 = cv2.morphologyEx(canny_img,cv2.MORPH_CLOSE , morhology_kernel)
        # morphology_img2 = cv2.morphologyEx(morphology_img1, cv2.MORPH_CLOSE, morhology_kernel)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        morphology_img1 = cv2.erode(thresh_img, element, iterations=1)
        #morphology_img1 = thresh_img
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 1))
        morphology_img2 = cv2.dilate(morphology_img1,  element,iterations = 1)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        morphology_img3 = cv2.erode(morphology_img2, element, iterations=1)
        #寻找轮廓（车牌矩形）
        outline_img , contours , hierarchy = cv2.findContours(morphology_img3.copy() ,cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE )
        #描绘所有轮廓
        print('全部轮廓个数：', len(contours))
        draw_alloutline_img = cv2.drawContours(origin_img.copy(),contours , -1 ,(0,0,255) , 1)
        #确定车牌矩形
        #面积筛选
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
        print('面积有效轮廓个数：', len(contours))
        #描绘面积筛选后剩下的轮廓
        draw_area_outline_img = cv2.drawContours(origin_img.copy(), contours, -1, (0, 0, 255), 1)
        draw_area_rect_img = origin_img.copy()
        draw_ratio_rect_img = origin_img.copy()
        # 描绘筛选面积后的的矩形
        ratiotrue_num = 0
        for i in range(len(contours)):
            cnt = contours[i]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #长宽比筛选
            height = math.sqrt(pow(abs(box[0,0]-box[1,0]),2)+pow(abs(box[0,1]-box[1,1]),2))
            width = math.sqrt(pow(abs(box[1, 0] - box[2, 0]), 2) + pow(abs(box[1, 1] - box[2, 1]), 2))
            ratio1 = height / width
            ratio2 = width / height
            draw_area_rect_img = cv2.drawContours(draw_area_rect_img, [box], 0, (0, 0, 255), 2)
            print("ratio1: ", ratio1 ,"ratio2: " , ratio2)
            #符合长宽比
            if (ratio1 > 2 and ratio1 < 5)or(ratio2 > 2 and ratio2 <6):
                draw_ratio_rect_img = cv2.drawContours(draw_ratio_rect_img, [box], 0, (0, 0, 255), 2)
                ratiotrue_num += 1
        print("长宽比有效矩形：" , ratiotrue_num)


        #显示已处理图像
        plt.figure("image processing"),plt.subplot(321),plt.imshow(cv2.cvtColor(origin_img,cv2.COLOR_BGR2RGB)),plt.title("origin image"),plt.axis("off")
        plt.figure("image processing"),plt.subplot(322),plt.imshow(cv2.cvtColor(gauss_img,cv2.COLOR_BGR2RGB)),plt.title("gaussianblur image"),plt.axis("off")
        plt.figure("image processing"), plt.subplot(323), plt.imshow(gray_gauss_img,"gray"), plt.title("gray_gauss_img"),plt.axis("off")
        plt.figure("image processing"), plt.subplot(324), plt.imshow(sobel_img, "gray"), plt.title("sobel_img"),plt.axis("off")
        plt.figure("image processing"), plt.subplot(325), plt.imshow(thresh_img, "gray"), plt.title("thresh_img"),plt.axis("off")
        plt.figure("image processing_2"), plt.subplot(331), plt.imshow(morphology_img1, "gray"), plt.title("morphology_img1"),plt.axis("off")
        plt.figure("image processing_2"), plt.subplot(332), plt.imshow(morphology_img2, "gray"), plt.title("morphology_img2"),plt.axis("off")
        plt.figure("image processing_2"), plt.subplot(333), plt.imshow(morphology_img3, "gray"), plt.title("morphology_img3"), plt.axis("off")
        plt.figure("image processing_2"), plt.subplot(334), plt.imshow(cv2.cvtColor(draw_alloutline_img,cv2.COLOR_BGR2RGB)), plt.title("draw_alloutline_img"),plt.axis("off")
        plt.figure("image processing_2"), plt.subplot(335), plt.imshow(cv2.cvtColor(draw_area_outline_img,cv2.COLOR_BGR2RGB)), plt.title("draw_area_outline_img"),plt.axis("off")
        plt.figure("image processing_2"), plt.subplot(336), plt.imshow(cv2.cvtColor(draw_area_rect_img, cv2.COLOR_BGR2RGB)), plt.title("draw_area_rect_img"),plt.axis("off")
        plt.figure("image processing_2"), plt.subplot(337), plt.imshow(cv2.cvtColor(draw_ratio_rect_img, cv2.COLOR_BGR2RGB)), plt.title("draw_ratio_rect_img"), plt.axis("off")
        plt.show()