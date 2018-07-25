import numpy as np
import cv2
import math

from display.glm import glm
from src.Generator.generator import *
from src.Generator.stack import *

data_label_paths = {3: "data/one_op/expressions.txt" }
            #5: "../data/two_ops/expressions.txt",
            #7: "../data/three_ops/expressions.txt"}

gen = Generator(data_labels_paths=data_label_paths, primitives=None)

sim = SimulateStack(max_len=5, canvas_shape=[64,64,64], draw_uniques=None)

def find_points(a):
    l = []
    for i in range(64):
        for j in range(64):
            for k in range(64):
                if a[i,j,k]== True:
                    l.append(glm.vec3(i,j,k))

    return l

def border_find_points(a):
    l = []
    for i in range(64):
        for j in range(64):
            for k in range(64):
                if a[i,j,k]== True:
                    l.append([i,j,k])


    # remove inner points , only keep surface points(cubes)
    remove_list = []
    for index, point in enumerate(l):
        if not [point[0]-1,point[1],point[2]] in l:
            continue
        if not [point[0]+1,point[1],point[2]] in l:
            continue
        if not [point[0],point[1]-1,point[2]] in l:
            continue
        if not [point[0],point[1]+1,point[2]] in l:
            continue
        if not [point[0],point[1],point[2]-1] in l:
            continue
        if not [point[0],point[1],point[2]+1] in l:
            continue

        remove_list.append(index)

    for index,i in enumerate(remove_list):
        del l[ i-index ]
    
    return l

# see from the top toward -z axis
def z_parrallel_projection(voxel, w , h ):
    img = np.ones([w,h],dtype=float)
    depth = np.zeros([w,h], dtype=int)

    w_ratio = w / 64.0
    h_ratio = h / 64.0

    for i in range(w):
        for j in range(h):
            xs = range(math.ceil(math.floor(i) * w_ratio) , math.floor(math.ceil(i) * w_ratio) + 1)
            ys = range(math.ceil(math.floor(j) * h_ratio) , math.floor(math.ceil(j) * h_ratio) + 1)

            z = 63
            while (z >= 0):
                count = 0
                for x in xs:
                    for y in ys:
                        if voxel[x, y ,z] == True :
                            count += 1

                if (count > 0):
                    img[i,j] = max(0, 1 - count / len(xs) * len(ys) )
                    break

                z -= 1
    return img

# return the transfer matrix of axis rotation
def axis_view_matrix(axis: glm.vec3):
    zaxis = axis.normalize()

    yaxis = glm.vec3(1,1,(- zaxis.x - zaxis.y )/zaxis.z)
    yaxis = yaxis.normalize()

    xaxis = yaxis.cross(zaxis)
    xaxis = xaxis.normalize()

    return glm.mat3(xaxis.x,xaxis.y,xaxis.z,
        yaxis.x,yaxis.y,yaxis.z,
        zaxis.x,zaxis.y,zaxis.z)

# see from the transfered top toward -z' axis
def axis_view_parallel_projection(voxel ,axis:glm.vec3, w, h):
    point_list = find_points(voxel)

    transfer_matrix = axis_view_matrix(axis)

    new_point_list = []
    for point in point_list:
        new_point_list.append(transfer_matrix * point)
    
    


with open(data_label_paths[3]) as data_file:
    expressions = data_file.readlines()
    
for index,exp in enumerate(expressions):
    program = gen.parse(exp)

    sim.get_all_primitives(gen.primitives)
    sim.generate_stack(program, if_primitives=True)
    voxel = sim.stack.get_items()[0]

    #point_list = find_points(voxel)

    #projection 
    img = z_parrallel_projection(voxel, 64 , 64)

    img_mask = img * 255
    img_mask = np.array(img_mask,dtype=int)
    #img_mask = cv2.merge(img_mask)
    cv2.imwrite('data/2D/z-parallel/'+ str(index) +'.jpg' , img_mask)
        
    print('finish processing pic '+str(index))



    
    

