import numpy as np
import math
import os
import time

OBJ_SIZE = 10
OBJ_NUM  = 5
X_SIZE = 43
Y_SIZE = 168

Pixel_Buffer = np.full((X_SIZE,Y_SIZE),".")
camera_position = np.array([2,-3,1])
scene=[]
class Obj_3d:
    def __init__(self):
        self.name = ""
        self.vertex = np.zeros((OBJ_SIZE,3))
        self.faces = np.zeros((OBJ_SIZE,4))
        self.position = np.zeros((3))
        self.vertex_position_rel_to_cam = np.zeros((OBJ_SIZE,3))


def Obj_position_from_camera ():

    for x in range(0,1):
        for y in range(0,OBJ_SIZE):
            for z in range(0,3):
                scene[x].vertex_position_rel_to_cam[y,z] = scene[x].position[z] + scene[x].vertex[y,z] - camera_position[z]

def transformation_3d_to_2d():
    for z in range(0,1):
        for x in range(0,OBJ_SIZE):
                if  (0<(abs(scene[z].vertex_position_rel_to_cam[x,0] / scene[z].vertex_position_rel_to_cam[x,2])) +X_SIZE/2 < X_SIZE) and (0<(abs( scene[z].vertex_position_rel_to_cam[x,1] / scene[z].vertex_position_rel_to_cam[x,2]))+Y_SIZE/2<Y_SIZE):

                    Pixel_Buffer[ round(scene[z].vertex_position_rel_to_cam[x,0] / (0.1 * scene[z].vertex_position_rel_to_cam[x,2]) + 20 ),round( scene[z].vertex_position_rel_to_cam[x,1] / (0.1*scene[z].vertex_position_rel_to_cam[x,2])+80) ]= "0"

def Render():

    while(1):

        for i in range(0,8):
            scene[0].vertex[i,0] +=1
        Obj_position_from_camera()
        transformation_3d_to_2d()
        print(''.join(''.join(str(cell) for cell in row) for row in Pixel_Buffer))

        time.sleep(.05)
        os.system("cls")
        for x in range(0,X_SIZE):
            for y in range(Y_SIZE):
                Pixel_Buffer[x,y] = "-"


def main():
    scene.append(Obj_3d())

    scene[0].vertex[0,0] = 50
    scene[0].vertex[1,0] = 50
    scene[0].vertex[2,0] = 50
    scene[0].vertex[3,0] = 50
    scene[0].vertex[4,0] = -50
    scene[0].vertex[5,0] = -50
    scene[0].vertex[6,0] = -50
    scene[0].vertex[7,0] = -50

    scene[0].vertex[0,1] = 50
    scene[0].vertex[1,1] = 50
    scene[0].vertex[2,1] = -50
    scene[0].vertex[3,1] = -50
    scene[0].vertex[4,1] = 50
    scene[0].vertex[5,1] = 50
    scene[0].vertex[6,1] = -50
    scene[0].vertex[7,1] = -50

    scene[0].vertex[0,2] = 150
    scene[0].vertex[1,2] = 50
    scene[0].vertex[2,2] = 150
    scene[0].vertex[3,2] = 50
    scene[0].vertex[4,2] = 150
    scene[0].vertex[5,2] = 50
    scene[0].vertex[6,2] = 150
    scene[0].vertex[7,2] = 50


    scene[0].position = [-5,3,50]

    Render()

main()