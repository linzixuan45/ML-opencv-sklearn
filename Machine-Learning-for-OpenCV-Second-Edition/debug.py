'''初始化'''
import numpy as np
import matplotlib.pyplot as plt
import cv2

'''Sober算子,初始化'''
sob_x, sob_y = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], \
               [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
'''图片读取并灰度化'''

# 类型type(img1_gray)  numpy.ndarray,shape (390, 700)

# plt.imshow(img1_gray, cmap='gray')
np.random.seed(7)
img = cv2.imread('figures/tiger.jpg')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.show()
'''定义一个卷积函数'''


def convolution(kernal, data_gray):
    # 没加padding, 最下面两行无法计算 ，故而需要减去2 ，类比于torch.nn 类中的 nn.Conv2d
    n, m = data_gray.shape  # 返回元组
    img_new = []
    for i in range(n - 3):
        line = []
        for j in range(m - 3):
            temp = data_gray[i:i + 3, j:j + 3]
            line.append(np.sum(np.multiply(kernal, temp)))
        img_new.append(line)
    return np.array(img_new)


def convolution_ndarray(kernal, data_gray):
    n, m = data_gray.shape  # (390, 700)
    img_new = np.zeros((n-2, m-2))
    for i in range(n-2 ):  # [0,387)
        temp_row = np.zeros(m-2)  # 700
        for j in range(m -2):  # [ 0,697)
            temp = data_gray[i:i + 3, j:j + 3]
            temp_row[j] = np.sum(np.multiply(kernal, temp))
        img_new[i] = temp_row
    return img_new


#Gx1 = convolution(sob_x, img1)
Gx = convolution_ndarray(sob_x, img1)

#Gy1 = convolution(sob_y, img1)
Gy = convolution_ndarray(sob_y, img1)

G_L1 = np.absolute(Gx) +np.absolute(Gy)
# L2范数
G_L2 = np.sqrt(pow(Gx,2)+pow(Gy,2))

# 255矩阵
ones = np.ones(G_L1.shape)*255 #ones.shape (387, 697)


plt.figure(figsize=(40,40))
plt.subplot(131)
plt.imshow(img1, cmap='gray')
plt.title('Sober + row_data ')

Color_Reversal_1 = ones -G_L1 #颜色反转
plt.subplot(132)
plt.imshow(Color_Reversal_1,cmap='gray')
plt.title(' Sober + G_L1')

Color_Reversal_2 = ones-G_L2    #颜色反转
plt.subplot(133)
plt.imshow(Color_Reversal_2,cmap='gray')
plt.title(' Sober +G_L2 ')
plt.show()
