import numpy as np
import cv2 as cv
import math

class ray:
    def __init__ (self,a,b):
        self.direction = b
        self.origin = a
    def point_at_t(self,t):
        return self.origin + self.direction * t


def intersection_sphere(raio):
    C = np.array([1,1,5])
    R2 = 1**2
    a = np.dot(raio.direction,raio.direction)
    b = 2*np.dot(raio.origin-C,raio.direction)
    c = np.dot(raio.origin-C,raio.origin-C) - R2

    delta = b**2 - 4*a*c

    if delta > 0:
        sun_direction = np.array([0.707,-0.707,0])
        t0 = (-b+math.sqrt(delta))/(2*a)
        intersection_point = raio.point_at_t(t0)
        normal = -1*(intersection_point-C)/np.dot(intersection_point-C,intersection_point-C)
        light_value = np.dot(normal,sun_direction)

        if light_value > 0:
            light_value = math.floor(255*abs(light_value))
        return np.array([0,0,light_value],dtype=np.uint8)
    else:
        return np.array([0,10,0],dtype=np.uint8)


width = 480
height = 480

image_array = np.zeros((height,width,3),dtype=np.uint8) 

for j in range(0,image_array.shape[0]):
    for i in range(0,image_array.shape[1]):
        image_array[j,i,0] = ((height - j)*100+j* 200)/height  
        image_array[j,i,1] = ((height - j)*50+j* 100)/height  
        raio = ray(np.array([0,0,0]),np.array([i/width -0.5,j/height-0.5,1]))
        bool_intersection = intersection_sphere(raio)
        if np.any(bool_intersection != np.array([0,10,0],dtype=np.uint8)):
            image_array[j,i,:] = bool_intersection

image_array = cv.flip(image_array,0)
cv.imshow("a",image_array)
cv.waitKey(0)