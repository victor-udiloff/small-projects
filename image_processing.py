import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Import picture
# pic is the original picture
# pic2 is the picture to be modified

pic = cv.imread("q.jpg")
pic2 = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)



# RESIZE AND INTERPOLATION
'''
g = 0.5
for i in range(0,pic.shape[0]):
    for j in range(0,pic.shape[1]):
         pic2[i,j] = np.power(  pic[i,j] /255  ,g) * 255

pic2 = cv.resize(pic2,(1000,1000), interpolation= cv.INTER_NEAREST)

cv.imshow("a",pic)
cv.imshow("b",pic2)
cv.waitKey(0)
'''



'''
#HISTOGRAM EQUALIZATION

pic_hist,b = np.histogram(pic2,bins=255)
pic_hist = np.transpose(pic_hist) / (pic2.shape[0]*pic2.shape[1])
pic_cumulative = np.zeros(256)

ramp = np.zeros(256)
rampc = np.zeros(256)
rampci = np.zeros(256)


for i in range(1,256):
    ramp[i] = ramp[i-1] + 1 
ramp = (ramp/255) * (2/255)

for i in range(0,256):
    pic_cumulative[i] = np.sum(pic_hist[0:i])
    rampc[i] = np.sum(ramp[0:i])


for i in range(0,256):
    rampci[round(255*rampc[i])] = i

    if i ==115:
        print(i)
        print(rampc[i])
        print(round(255*rampc[i]))



pic_cumulative = 255* pic_cumulative

for i in range(0,pic2.shape[0]):
    for j in range(0,pic2.shape[1]):
        pic2[i,j] = pic_cumulative[pic2[i,j]]
        pic2[i,j] = rampci[pic2[i,j]]



cv.imshow("a",pic)
cv.imshow("b",pic2)
cv.waitKey(0)

a22,b22 = np.histogram(pic2,bins=255)
a22 = np.transpose(a22) / (pic2.shape[0]*pic2.shape[1])

'''

'''
# See noise 

#plt.plot(a22)
#plt.show()


N = 50
x_centers = np.round(np.linspace(0,pic2.shape[1],N))
y_centers = np.round(np.linspace(0,pic2.shape[0],N))

#x_gradient = cv.subtract(pic2,np.roll(pic2,1,axis=1))
#y_gradient = cv.subtract(pic2,np.roll(pic2,1,axis=1))
#abs_gradient = np.round(np.sqrt( (np.power(x_gradient,2)) + np.power(y_gradient,2) )).astype(np.uint8)

sobel_filter_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_filter_y  = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
laplacian_filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
 

x_gradient = cv.filter2D(src=pic2,kernel=sobel_filter_x,ddepth=-1)
y_gradient = cv.filter2D(src=pic2,kernel=sobel_filter_y,ddepth=-1)
abs_gradient = np.round(np.sqrt( (np.power(x_gradient,2)) + np.power(y_gradient,2) )).astype(np.uint8)

plt.hist(pic2[20:40,620:640])
plt.show()
#cv.imshow("a",pic2[20:40,620:640])
#cv.waitKey(0)
#print((pic2>50).astype(np.uint8))
'''


#Spread spectrum watermarking

K = 100

s = np.random.normal(0,2,K)

fft_pic = np.fft.fft2(pic2)
fft_pic2 = np.copy(fft_pic)

for i in range(0,K):
    x = np.max(fft_pic2)
    a =  np.where(fft_pic2 == x)
    print(a) 
    print(x) 
