import cv2
import json
from matplotlib import pyplot as plt
from pylab import mpl         #改matplotlib显示字体格式
import numpy as np
from numpy.linalg import norm
import math
import operator as op
import sys
import os

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体（仿宋）


SZ = 20
PROVINCE_START = 1000


# 来自opencv的sample，用于svm训练
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

#不能保证包括所有省份
provinces = [
"zh_cuan", "川",
"zh_e", "鄂",
"zh_gan", "赣",
"zh_gan1", "甘",
"zh_gui", "贵",
"zh_gui1", "桂",
"zh_hei", "黑",
"zh_hu", "沪",
"zh_ji", "冀",
"zh_jin", "津",
"zh_jing", "京",
"zh_jl", "吉",
"zh_liao", "辽",
"zh_lu", "鲁",
"zh_meng", "蒙",
"zh_min", "闽",
"zh_ning", "宁",
"zh_qing", "靑",
"zh_qiong", "琼",
"zh_shan", "陕",
"zh_su", "苏",
"zh_sx", "晋",
"zh_wan", "皖",
"zh_xiang", "湘",
"zh_xin", "新",
"zh_yu", "豫",
"zh_yu1", "渝",
"zh_yue", "粤",
"zh_yun", "云",
"zh_zang", "藏",
"zh_zhe", "浙"
]
class StatModel(object):
	def load(self, fn):
		self.model = self.model.load(fn)
	def save(self, fn):
		self.model.save(fn)
class SVM(StatModel):
	def __init__(self, C = 1, gamma = 0.5):
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(C)
		self.model.setKernel(cv2.ml.SVM_RBF)
		self.model.setType(cv2.ml.SVM_C_SVC)
#训练svm
	def train(self, samples, responses):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
#字符识别
	def predict(self, samples):
		r = self.model.predict(samples)
		return r[1].ravel()


#车牌对象
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

    def train_svm(self):
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("train\\chars2"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(root_int)

            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.model.train(chars_train, chars_label)
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")
        else:
            chars_train = []
            chars_label = []
            for root, dirs, files in os.walk("train\\charsChinese"):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                index = provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(index)
            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.modelchinese.train(chars_train, chars_label)

    def save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")
        if not os.path.exists("svmchinese.dat"):
            self.modelchinese.save("svmchinese.dat")


    class License_rect:

        def __init__(self, img, color, limits):
            self.img = img
            self.color = color
            self.limits = limits

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

    #颜色筛选
    def color_filter(self, img ,index , blue_lower ,blue_upper , green_lower , green_upper):
        HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_S = blue_lower[1]
        green_S = green_lower[1]
        green_pixelpoint = blue_pixelpoint = 0
        # 获取图片大小（行列像素点）
        row, col = img.shape[0:2]
        all_pixelpoint = row * col
        # print("shape:" , row, col)
        # 遍历像素点，分析像素点颜色
        # 整行遍历
        colors = []
        for r, row_HSV_point in enumerate(HSV_img):
            # 行 ---> 点遍历
            for c, HSV_point in enumerate(row_HSV_point):
                H = HSV_point[0]
                S = HSV_point[1]
                V = HSV_point[2]
                # print("HSV:",H , S ,V)
                limit_min = 0
                limit_max = 255
                if (35 < H <= 77) and (S > green_S):
                    green_pixelpoint += 1
                if (100 < H <= 124) and (S > blue_S):
                    blue_pixelpoint += 1
        # print("绿色占比：{:.2f}".format(green_pixelpoint / all_pixelpoint),
        #       "蓝色占比：{:.2f}".format(blue_pixelpoint / all_pixelpoint))
        if green_pixelpoint >= blue_pixelpoint:
            if (green_pixelpoint * 5.1 >= all_pixelpoint):
                colors.append("green")
                #对于绿色车牌，将黄色以及蓝色 H 范围包括
                limit_min = 26
                limit_max = 99
                # print("该矩形绿色居多")
        elif (blue_pixelpoint * 5.1 >= all_pixelpoint):
                colors.append("blue")
                # 对于蓝色车牌，将黄色以及蓝色 H 范围包括
                limit_min = 78
                limit_max = 155
                # print("该矩形蓝色居多")
        #是否符合颜色筛选
        if colors:
               return True ,img ,  colors ,(limit_min,limit_max)
        else :
               return False ,img , colors,(limit_min,limit_max)

    def point_limit(self , point):
        if point[0] < 0:
            point[0] = 0
        if point[1] < 0:
            point[1] = 0

    def cut_palte(self , license_rect):
        img = license_rect.img
        HSV_img = cv2.cvtColor( img , cv2.COLOR_BGR2HSV )
        color = license_rect.color[0]
        img_row , img_col = HSV_img.shape[0:2]
        x_left = img_col
        x_right = 0
        y_top = img_row
        y_bottom = 0
        row_limit = 3
        col_limit = 30
        #由于绿色车牌渐变绿的特点，不进行 y_top 方向裁剪
        if op.eq( color , 'green'  ) == True:
            row_limit = 1
            col_limit = 30
            y_top = 0
        print("{}".format(license_rect.color) , "img_shape" ,img.shape[0:2] )
        for row in range(img_row):
            # 行 ---> 点遍历
            count = 0
            for col in range(img_col):
                H = HSV_img.item( row , col , 0)
                S = HSV_img.item( row , col , 1)
                V = HSV_img.item( row , col , 2)
                # print("HSV:",H , S ,V)
                # print(license_rect.limits[0],H,license_rect.limits[1],S)
                if ( license_rect.limits[0] < H <license_rect.limits[1] ) :
                    count += 1
            #y_bottom 标定车牌边界下沿
            #y_top    标定车牌边界上沿
            # print(count)
            if count >= col_limit :
                if y_bottom <= row :
                    y_bottom = row
                if y_top >= row :
                    y_top = row
        for col in range( img_col ):
            # 列 ---> 点遍历
            count = 0
            for row in range( img_row ):
                H = HSV_img.item( row , col , 0)
                S = HSV_img.item( row , col , 1)
                V = HSV_img.item( row , col , 2)
                # print("HSV:",H , S ,V)
                if license_rect.limits[0] < H <license_rect.limits[1] and S >60 :
                    count += 1
            #y_bottom 标定车牌边界下沿
            #y_top    标定车牌边界上沿
            # print(count)
            if count > row_limit :
                if x_left > col :
                    x_left = col
                if x_right < col :
                    x_right = col
        # print("y_top" , y_top , "y_bottom: " ,y_bottom )
        # print("x_left", x_left, "x_right: ", x_right)
        return y_top , y_bottom , x_left , x_right
    #寻找波峰
     #判断大于阈值的第一个点
     #寻找小于阈值并远离起始点两个
    def find_wave(self , threshold , histogram ):
        start_point = -1
        peak = False
        #判断前边界是否为起始点
        if histogram[0] > threshold :
              start_point = 0
              peak = True
        peaks_area = []
        for index , value in enumerate( histogram ) :
            #未处于峰值区域，寻找下一个峰值起始点
            if not peak and value >= threshold :
                   start_point = index
                   peak = True
            #若处于峰值区域，判断峰值区域大小
            if peak and value < threshold :
                   if index - start_point > 2 :
                           end_point = index
                           peaks_area.append( (start_point ,end_point ) )
                           peak = False
        ##判断后边界是否为终点
        if peak and index - start_point > 4 :
            peaks_area.append((start_point, index))
        return peaks_area


    def peaks_filter(self , peaks , spacing_limit , num_limit):
        if len(peaks) > num_limit :
            avrage_width = 0
            # 宽度均值滤波（去掉最大最小值,但当作peaks总数目不变处理）
            peaks_max = max(peaks, key=lambda item: item[1] - item[0])
            peaks_width_max = peaks_max[1] - peaks_max[0]
            peaks_min = min(peaks, key=lambda item: item[1] - item[0])
            peaks_width_min = peaks_min[1] - peaks_min[0]
            for peak in peaks:
                avrage_width += peak[1] - peak[0]
                # print(peak[1] - peak[0])
            avrage_width = (avrage_width - peaks_width_max - peaks_width_min) / (len(peaks))
            print("avrage:", avrage_width)
            peaks_copy = peaks.copy()
            #波峰筛选
             #波峰宽度筛选
             #波峰邻距筛选
            for item in range(len(peaks_copy)):
                peak_width = peaks_copy[item][1] - peaks_copy[item][0]
                if peak_width < avrage_width:
                    # (0 , 0)代表不和相邻的波峰合并
                    # (0 , 1)代表和相邻的右侧波峰合并
                    # (1 , 0)代表和相邻的做侧波峰合并
                    peaks[item] = (0, 0)
                    # 判断是否为左右侧波峰
                    # if   item == 0  :
                    #         if (peaks_area_copy[item + 1][0] - peaks_area_copy[item][1]) > spacing_limit :
                    #             pass
                    # elif item == len(peaks_area_copy ) -1 :
                    #         if (peaks_area_copy[item][0] - peaks_area_copy[item - 1][1]) > spacing_limit :
                    #             pass
                    #
                    # elif (peaks_area_copy[item ][0] - peaks_area_copy[item - 1][1])> spacing_limit :
                    #             # 在原波峰组中删除该波峰
                    #             peaks_area[item][0] = (1, 0)
            item = 0
            while (1):
                if item > len(peaks) - 1:
                    break
                if peaks[item] == (0, 0):
                    peaks.pop(item)
                elif peaks[item] != (0, 0):
                    item += 1
            print("peaks_width_filter : peaks :" , peaks , "num : " , len(peaks))
            if len(peaks) > num_limit :
                spacings = []
                # 间距最值滤波
                for index in range( len( peaks ) - 1  ):
                    spacings.append(peaks[index + 1][0] - peaks[index][1] )
                    print(peaks[index + 1][0] - peaks[index][1])
                #间距干扰峰波通常在两端
                 #最大间距通常为第二个字符与第三个字符之间的距离
                spacing_max = max(spacings)
                print( spacing_max == spacings[0] )
                while spacing_max != spacings[1] :
                    spacing_max_L = spacing_max
                    if spacing_max == spacings[1] :
                            break
                    elif spacing_max == spacings[0] :
                            spacings.pop(0)
                            peaks.pop(0)
                    elif spacing_max == spacings[-1] :
                            spacings.pop(-1)
                            peaks.pop(-1)
                    spacing_max = max(spacings)
                    #与之前的间距一样
                    if spacing_max == spacing_max_L :
                        break
                print("peaks_spacing_filter : peaks :" , peaks , "num : " , len(peaks))
            return peaks
        else :
            return peaks








    def find_carplate(self , img):

        origin_img = img.copy()
        pic_hight, pic_width = origin_img.shape[:2]
        #高斯模糊处理
        gauss_img = cv2.GaussianBlur(origin_img.copy(), (self.gaussian_blur, self.gaussian_blur), 0)
        # 图像特定颜色区域提取（蓝色与绿色）
        color_mask_img = self.color_mask(gauss_img)
        #中值滤波
        media_blue_img = cv2.medianBlur(color_mask_img,3)
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
        ratiotrue_rect = []
        for i in range(len(contours)):
            cnt = contours[i]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #长宽比筛选
              #通过判断box[0] 与中心点的水平位置关系判断车牌左边上倾，还是右边上倾
            #左边上倾，矩形右下角为bow[0]
            if rect[0][0] < float(box[0][0]):
                print("车牌可能左边上倾...")
                width = math.sqrt(pow(abs(box[0, 0] - box[1, 0]), 2) + pow(abs(box[0, 1] - box[1, 1]), 2))
                height = math.sqrt(pow(abs(box[1, 0] - box[2, 0]), 2) + pow(abs(box[1, 1] - box[2, 1]), 2))
            if rect[0][0] > float(box[0][0]):
                print("车牌可能右边上倾...")
                height = math.sqrt(pow(abs(box[0,0]-box[1,0]),2)+pow(abs(box[0,1]-box[1,1]),2))
                width = math.sqrt(pow(abs(box[1, 0] - box[2, 0]), 2) + pow(abs(box[1, 1] - box[2, 1]), 2))
            ratio =  width / height
            draw_area_rect_img = cv2.drawContours(draw_area_rect_img, [box], 0, (0, 0, 255), 2)
            #符合长宽比
            if (ratio > 2 and ratio <6):
                draw_ratio_rect_img = cv2.drawContours(draw_ratio_rect_img, [box], 0, (0, 0, 255), 2)
                ratiotrue_rect.append(rect)
                ratiotrue_num += 1
                # print("width:", width);
                # print("height:", height)
                # print("ratio: ", ratio)
        #判断是否寻找到长宽比有效矩形
        if  not ratiotrue_num:
            print("未找到长宽比有效矩形....")

        elif ratiotrue_num > 0:
            print("找到   {}  个长宽比有效矩形".format(ratiotrue_num))
            #获取有效长宽比有效矩形区域image并进行角度变换
            # for rect in ratiotrue_rect:
            #     print(rect)
            #     box = cv2.boxPoints(rect)
            #     box = np.int0(box)
            #     temp_img = cv2.drawContours(temp_img, [box], 0, (0, 0, 255), 2)
            cardplate_imgs = []
            temp = 1
            for rect in ratiotrue_rect:
                print(rect)
                #对于矩形旋转角度为0，改变角度，放置点重叠
                if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
                    angle = 1
                    print("创造角度")
                else:
                    angle = rect[2]
                    print("角度不变")
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # width(box[0] 、box[3] 距离) ：rect[1][0]
                # height(box[0] 、 box[1] 距离) ：rect[1][1]
                if rect[0][0] < float(box[0][0]):
                    #print("车牌可能左边上倾...")
                    rect = (rect[0], (rect[1][0] + 10, rect[1][1] + 15), angle)  # 扩大范围，避免车牌边缘被排除
                if rect[0][0] > float(box[0][0]):
                    #print("车牌可能右边上倾...")
                    rect = (rect[0], (rect[1][0] + 10, rect[1][1] + 10), angle)  # 扩大范围，避免车牌边缘被排除

                box = cv2.boxPoints(rect)
                heigth_point = right_point = [0, 0]
                left_point = low_point = [pic_width, pic_hight]
                #将四个点指向最大值点 #点可能重复
                 # left_point ----> X最小
                 # right_point ----> X最大
                 # low_point ----> Y最小
                 # height_point ----> Y最大
                for point in box:
                    if left_point[0] > point[0]:
                        left_point = point
                    if low_point[1] > point[1]:
                        low_point = point
                    if heigth_point[1] < point[1]:
                        heigth_point = point
                    if right_point[0] < point[0]:
                        right_point = point

                if left_point[1] <= right_point[1]:  # 正角度
                    new_right_point = [right_point[0], heigth_point[1]]
                    pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
                    pts1 = np.float32([left_point, heigth_point, right_point])
                    M = cv2.getAffineTransform(pts1, pts2)
                    dst = cv2.warpAffine(origin_img.copy(), M, (pic_width, pic_hight))
                    self.point_limit(new_right_point)
                    self.point_limit(heigth_point)
                    self.point_limit(left_point)
                    card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
                    cardplate_imgs.append(card_img)
                    # plt.figure("矩形仿射变换"),plt.subplot(3 , 2 , temp),plt.imshow(cv2.cvtColor(card_img,cv2.COLOR_BGR2RGB))
                    # temp += 1

                elif left_point[1] > right_point[1]:  # 负角度

                    new_left_point = [left_point[0], heigth_point[1]]
                    pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
                    pts1 = np.float32([left_point, heigth_point, right_point])
                    M = cv2.getAffineTransform(pts1, pts2)
                    dst = cv2.warpAffine(origin_img.copy(), M, (pic_width, pic_hight))
                    self.point_limit(right_point)
                    self.point_limit(heigth_point)
                    self.point_limit(new_left_point)
                    card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
                    cardplate_imgs.append(card_img)
                    # plt.figure("矩形仿射变换"), plt.subplot(3, 2, temp), plt.imshow(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB))
                    # temp += 1
            blue_lower  = np.array([100, 200, 46])
            blue_upper  = np.array([124, 255, 255])
            green_lower = np.array([34, 200, 46])
            green_upper = np.array([77, 255, 255])
            blue_S  = blue_lower[1]
            green_S = green_lower[1]
            onemore_colorfilter_img = False
            print("blue_S init:", blue_S, "green_S init", green_S)
            #颜色定位，并进行筛选（蓝底、绿底）
            license_rects = []
            ###################################################
            ## 使用这种递减的办法无法处理图片中出现多车牌的情况
            ###################################################
            while blue_S > 60 or green_S > 60:
                for index , maycardplate_img in enumerate(cardplate_imgs) :
                        T_or_F , colorfilter_img , carplate_color ,( limit_min , limit_max ) = self.color_filter(maycardplate_img ,
                                                                                                                           index , blue_lower ,blue_upper , green_lower , green_upper ,)
                        if T_or_F ==True :
                              onemore_colorfilter_img = True
                              license_rects.append(self.License_rect( colorfilter_img, carplate_color , ( limit_min , limit_max ) ))
                              # plt.figure("符合颜色"), plt.subplot(3, 2, index + 1), plt.imshow(cv2.cvtColor(colorfilter_img, cv2.COLOR_BGR2RGB))
                              ######################################################
                              # 将图片根据颜色扣取的部分可视化
                              # 调节色调和、饱和度 #车牌饱和度通常较大
                              # blue_mask = cv2.inRange(HSV_img, blue_lower, blue_upper)
                              # green_mask = cv2.inRange(HSV_img, green_lower, green_upper)
                              # color_mask = cv2.bitwise_or(blue_mask, green_mask)
                              # color_mask_img = cv2.bitwise_and(maycardplate_img, maycardplate_img, mask=color_mask)
                              # plt.figure("color_fiter"), plt.subplot(3, 2, index + 1), plt.imshow(cv2.cvtColor(color_mask_img, cv2.COLOR_BGR2RGB))
                              #######################################################
                        elif T_or_F ==False:
                              pass
                #若有图片符合颜色筛选，退出while
                if   onemore_colorfilter_img == True :
                       print(print("blue_S final:", blue_S, "green_S final", green_S))
                       break
                elif onemore_colorfilter_img ==False :
                    #未有图片符合颜色筛选，配置饱和度重新筛选
                    blue_lower[1] -= 1
                    green_lower[1] -= 1
                    blue_S = blue_lower[1]
                    green_S = green_lower[1]
            if onemore_colorfilter_img == True :
                  #裁剪车牌，去除非车牌边界
                  for index , license_rect in enumerate(license_rects):
                      # plt.figure("符合颜色筛选的矩形"), plt.subplot(3, 2, index + ???), plt.imshow(cv2.cvtColor(license_rect.img, cv2.COLOR_BGR2RGB)), plt.title("符合颜色的矩形")
                      y_top, y_bottom, x_left, x_right = self.cut_palte(license_rect)
                      license_rect.img = license_rect.img[ y_top : y_bottom , x_left : x_right ]
                      # plt.figure("车牌矩形"), plt.subplot(2,1, index +1 ), plt.imshow(cv2.cvtColor(license_rect.img, cv2.COLOR_BGR2RGB)),plt.title("定位到的车牌")
                  #车牌字符分割
                  for index, license_rect in enumerate(license_rects):
                      img = license_rect.img
                      color = license_rect.color[0]
                      gray_img = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)

                      plt.figure("车牌字符分割"), plt.subplot(3, 3, index + 1), plt.imshow(gray_img, "gray"),plt.title("gray_img")
                      ret , thresh_img = cv2.threshold( gray_img ,0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                      #将二值化图像都转化为黑底白字
                      if op.eq( color , "green" ) == True :
                          thresh_img = cv2.bitwise_not(thresh_img , thresh_img)
                      plt.figure("车牌字符分割"), plt.subplot(3, 3, index + 2), plt.imshow(thresh_img, "gray"), plt.title("thresh_img")
                      #查找水平直方图波峰
                        #判断并裁减车牌的高度
                      #在 Y 方向进行腐蚀
                      element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
                      y_erode_img = cv2.erode( thresh_img ,element , iterations = 1  )
                      plt.figure("车牌字符分割"), plt.subplot(3, 3, index + 3), plt.imshow(y_erode_img, "gray"), plt.title("Y 方向腐蚀")
                      x_histogram = np.sum(y_erode_img , axis = 1) #压缩成一列
                      #绘制水平方向直方图
                      # for i , value in enumerate( x_histogram ) :
                      #      plt.figure("水平方向直方图"),plt.bar( i , value ,width = 1 , facecolor = "black" ,edgecolor = "white")
                      x_min = np.min( x_histogram )
                      x_max = np.max( x_histogram )
                      #去掉最大最小，过滤异常波峰
                      x_average = (np.sum( x_histogram ) - x_min - x_max ) / x_histogram.shape[0]
                      x_threshold = ( x_min + x_average ) / 2
                      peaks = self.find_wave(x_threshold , x_histogram )
                      # print(peaks_area)
                      peaks_width_max = max(peaks , key = lambda item : item[1] - item[0] )
                      # print(peaks_area_max)
                      thresh_img_y = thresh_img[ peaks_width_max[0] : peaks_width_max[1] ]
                      plt.figure("车牌字符分割"), plt.subplot(3, 3, index + 4), plt.imshow(thresh_img_y, "gray"), plt.title("Y方向分割字符完成")
                      #查找垂直方向直方图波峰
                      # row_img , col_img = thresh_img.shape[0:2]
                      # thresh_img = thresh_img[1 : row_img -1 ]
                      # #防止白边影响阈值判断
                      # plt.figure("车牌字符分割"), plt.subplot(3, 2, index + ？？？), plt.imshow(thresh_img, "gray"), plt.title("顶部底部各裁一行")
                      # 在 XY 方向进行腐蚀
                      element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
                      xy_erode_img = cv2.erode(thresh_img_y, element, iterations=1)
                      element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                      xy_erode_img = cv2.erode(xy_erode_img, element, iterations=1)
                      plt.figure("车牌字符分割"), plt.subplot(3, 3, index + 5), plt.imshow(xy_erode_img, "gray"), plt.title("XY 方向同时腐蚀")
                      #获得可允许的波峰之间的最小距离
                      spacing_limit = xy_erode_img.shape[1] / 100
                      print("spacing_limit : " , spacing_limit )
                      y_histogram = np.sum(xy_erode_img, axis=0)  # 压缩成一列
                      for index2, value in enumerate(y_histogram):
                          plt.figure("垂直方向直方图"), plt.bar(index2, value, width=1, facecolor="black", edgecolor="white")
                      y_min = np.min(y_histogram)
                      y_max = np.max(y_histogram)
                      # 去掉最大最小，过滤异常波峰
                      y_average = (np.sum(y_histogram) - y_min - y_max) / y_histogram.shape[0]
                      y_threshold = (y_min + y_average) / 5
                      peaks = self.find_wave(y_threshold, y_histogram)
                      print(peaks)
                      peaks = self.peaks_filter(peaks , spacing_limit , 7)
                      # if len(peaks) != 7 :
                      #     print("404 : 字符个数不符合要求 ，寻找车牌字符失败......")
                      # else :
                      plate_num_pieces = []
                      for item in range(len(peaks) ) :
                          plate_num_pieces.append( thresh_img_y[ 0 : thresh_img_y.shape[0] ,peaks[item][0] : peaks[item][1] ])
                      for item , num_piece in enumerate(plate_num_pieces) :
                          # plt.figure("车牌字符分割结果"),plt.subplot(3 , 3 , item + 1) , plt.imshow(num_piece , "gray")
                          #计算两边需要扩展的长度，垂直方向不需要扩展，使sample同等大小
                          w = abs(num_piece.shape[1] - SZ) // 2
                          num_piece = cv2.copyMakeBorder(num_piece, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                          num_piece = cv2.resize(num_piece, (SZ, SZ), interpolation=cv2.INTER_AREA)
                          plt.figure("字符调整大小"), plt.subplot(3, 3, item + 1), plt.imshow(num_piece, "gray")
                          num_piece = preprocess_hog([num_piece])
                          # print(type(num_piece))
                          plt.figure("preprocess_hog"), plt.subplot(3, 3, item + 1), plt.imshow(num_piece, "gray")
                          if item == 0:
                              resp = self.modelchinese.predict(num_piece)
                              charactor = provinces[int(resp[0] - PROVINCE_START )]
                          else:
                              resp = self.model.predict(num_piece)
                              charactor = chr(resp[0])
                          print("resp",resp[0])
                          print(charactor)



        #显示已处理图像
        # plt.figure("image processing"),plt.subplot(321),plt.imshow(cv2.cvtColor(origin_img,cv2.COLOR_BGR2RGB)),plt.title("origin image"),plt.axis("off")
        # plt.figure("image processing"),plt.subplot(322),plt.imshow(cv2.cvtColor(gauss_img,cv2.COLOR_BGR2RGB)),plt.title("gaussianblur image"),plt.axis("off")
        # plt.figure("image processing"), plt.subplot(323), plt.imshow(gray_gauss_img,"gray"), plt.title("gray_gauss_img"),plt.axis("off")
        # plt.figure("image processing"), plt.subplot(324), plt.imshow(sobel_img, "gray"), plt.title("sobel_img"),plt.axis("off")
        # plt.figure("image processing"), plt.subplot(325), plt.imshow(thresh_img, "gray"), plt.title("thresh_img"),plt.axis("off")
        # plt.figure("image processing_2"), plt.subplot(331), plt.imshow(morphology_img1, "gray"), plt.title("morphology_img1"),plt.axis("off")
        # plt.figure("image processing_2"), plt.subplot(332), plt.imshow(morphology_img2, "gray"), plt.title("morphology_img2"),plt.axis("off")
        # plt.figure("image processing_2"), plt.subplot(333), plt.imshow(morphology_img3, "gray"), plt.title("morphology_img3"), plt.axis("off")
        #plt.figure("image processing_2"), plt.subplot(334), plt.imshow(cv2.cvtColor(draw_alloutline_img,cv2.COLOR_BGR2RGB)), plt.title("draw_alloutline_img"),plt.axis("off")
        # plt.figure("image processing_2"), plt.subplot(335), plt.imshow(cv2.cvtColor(draw_area_outline_img,cv2.COLOR_BGR2RGB)), plt.title("draw_area_outline_img"),plt.axis("off")
        # plt.figure("image processing_2"), plt.subplot(336), plt.imshow(cv2.cvtColor(draw_area_rect_img, cv2.COLOR_BGR2RGB)), plt.title("draw_area_rect_img"),plt.axis("off")
        # plt.figure("ratiotrue_rect"), plt.subplot(111), plt.imshow(cv2.cvtColor(draw_ratio_rect_img, cv2.COLOR_BGR2RGB)), plt.title("draw_ratio_rect_img"), plt.axis("off")
        plt.show()
