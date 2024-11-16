import cv2
import math
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('../data/new0020.png', cv2.IMREAD_GRAYSCALE)
    H, W = img.shape[0], img.shape[1]
    x = math.ceil(H / 2)
    y = math.ceil(W / 2)
    # 绘制两条直线，形成一个夹角
    img = cv2.line(img, (y, x), (W-1, x), (255, 255, 255), thickness=1, lineType=4)
    img = cv2.line(img, (y, x), (x, W-1), (255, 255, 255), thickness=1, lineType=4)

    cv2.imwrite('../data/line_0020.png', img)
    print(img.shape)


    # 转极坐标
    # 极坐标的高度
    theta = math.ceil(W / 2)
    # 极坐标的宽
    r = math.ceil(math.sqrt(math.ceil(H / 2) ** 2 + math.ceil(W / 2) ** 2))


    polar_img = np.zeros((theta, r))
    for tt in range(theta):
        for rr in range(r):
            j = round(x + rr * math.sin(tt / 180 * math.pi))
            i = round(y - rr * math.cos(tt / 180 * math.pi))
            if 0 <= i < H and 0 <= j < W:
                polar_img[tt, rr] = img[i, j]

    cv2.imwrite('../data/polar_0020.png', polar_img)