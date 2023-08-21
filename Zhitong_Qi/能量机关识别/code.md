#导入相关的库
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#读入视频
cap = cv.VideoCapture("target.MP4")

while True:
    #将视频一帧一帧保存
    ret, frame = cap.read()
    #调整视频窗口大小
    frame = cv.resize(frame,None,fx=0.5, fy=0.5)

    #转化为灰度
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #二值化
    ret,thresh = cv.threshold(gray,200,255,cv.THRESH_BINARY_INV)
    
    #漫水法
    seed_point = (100, 100)
    # 定义填充颜色
    fill_color = (255, 0, 0)
    # 创建掩膜图像，全黑
    mask = np.zeros((thresh.shape[0] + 2, thresh.shape[1] + 2), np.uint8)
    # 执行漫水法
    cv.floodFill(thresh, mask, seed_point, fill_color)

    # 创建结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    # 执行闭运算
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    #找出轮廓
    contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # 计算轮廓的面积
        area = cv.contourArea(contour)
        # 剔除过小的轮廓
        if area < 40 :
            # 绘制轮廓
            cv.drawContours(frame, [contour], -1, (0, 0, 255), 2)
    #显示视频
    cv.imshow('gray', gray)
    cv.imshow('thresh', thresh)
    cv.imshow('res', frame)
    if cv.waitKey(20) == 27:
        break

cap.release()

cv.destroyAllWindows()