import numpy as np
import cv2
import math

from display.glm import glm
from src.Generator.generator import *
from src.Generator.stack import *

data_label_paths = {3: "data/one_op/expressions.txt",
            5: "../data/two_ops/expressions.txt",
            7: "../data/three_ops/expressions.txt"}

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
                    # remove inner points , only keep surface points(cubes)
                    if (i > 0 and a[i-1,j,k] == True) and (i < 63 and a[i+1,j,k] == True) \
                        and (j > 0 and a[i,j-1,k] == True) and (j < 63 and a[i,j+1,k] == True) \
                        and (k > 0 and a[i,j,k-1] == True) and (k < 63 and a[i,j,k+1] == True):
                        continue
                    else:
                        l.append(glm.vec3(i,j,k))

    return l

# see from the top toward -z axis
def z_parrallel_projection(voxel, w , h ):
    img = np.ones([w,h],dtype=float)

    w_ratio = 64.0 / w
    h_ratio = 64.0 / h 

    # (0, 0) -> (0, 0)
    for i in range(w):
        for j in range(h):
            x_min = math.floor(i + 0.001) * w_ratio
            x_max = math.ceil(i + 0.001) * w_ratio
            y_min = math.floor(j + 0.001) * h_ratio
            y_max = math.ceil(j + 0.001) * h_ratio

            #xs = range(math.ceil(math.floor(i + 0.001) * w_ratio) , math.floor(math.ceil(i + 0.001) * w_ratio) )
            #ys = range(math.ceil(math.floor(j + 0.001) * h_ratio) , math.floor(math.ceil(j + 0.001) * h_ratio) )

            z = 63
            while (z >= 0):
                count = 0
                
                x = math.ceil(x_min)
                while(x < x_max):
                    y = math.ceil(y_min)
                    while(y < y_max):

                        if voxel[x, y ,z] == True :
                            count += 1
                            
                        y = y + 1
                    x = x + 1

                if (count > 0):
                    #img[i,j] = max(0, 1 - count / len(xs) * len(ys) )
                    img[i,j] = 0
                    break

                z -= 1
    return img


# origin_w : x side of the view
# origin_h : y side of the view
def z_parrallel_projection_point(point_list, origin_w , origin_h , w, h , center_x , center_y ):
    img = np.ones([w,h],dtype=float)
    #depth = np.zeros([w,h], dtype=float)

    point_index = [ [ [] for i in range(origin_h) ] for j in range(origin_w)  ] 
    
    # clean and then index the points 
    # may have negative index on p_x or p_y, but this doesn't matter. We refer to those elements by the same
    # negative index
    for point in point_list:
        p_x = math.floor(point[0])
        p_y = math.floor(point[1])
        p_z = point[2]

        p_x_ceiling = math.ceil(point[0])
        p_y_ceiling = math.ceil(point[1])

        point_index[p_x][p_y].append(p_z)
        if (p_x_ceiling != p_x):
            point_index[p_x_ceiling][p_y].append(p_z)
        if (p_y_ceiling != p_y):
            point_index[p_x][p_y_ceiling].append(p_z)
        if (p_x_ceiling != p_x and p_y_ceiling != p_y):
            point_index[p_x_ceiling][p_y_ceiling].append(p_z)


    w_ratio = origin_w / w
    h_ratio = origin_h / h


    # (center_x, center_y) -> (w/2, h/2)
    for i in range(w):
        for j in range(h):
            x_min = math.floor(i - w/2 + 0.001) * w_ratio + center_x
            x_max = math.ceil(i - w/2 + 0.001) * w_ratio + center_x
            y_min = math.floor(j - h/2 + 0.001) * h_ratio + center_y
            y_max = math.ceil(j - h/2 + 0.001) * h_ratio + center_y

            x = math.ceil(x_min)
            while( x < x_max ):
                y = math.ceil(y_min)
                while( y < y_max ):
                    points = point_index[x][y]

                    if points != []:
                        img[i][j] = 0

                    #for point_z in points:
                        # if point_z >= depth[i][j] :
                        #     if (point_z > depth[i][j]):
                        #         img[i][j] = 1
                        #         depth[i][j] = point_z
                            
                        #     img[i][j] = max( 0, img[i][j] - 1.0 / len(xs) * len(ys) )
                    
                    y = y + 1

                x = x + 1

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
# axis : the new z axis under the previous coordinate
def axis_view_place_points(voxel ,transfer_matrix):
    point_list = border_find_points(voxel)

    #axis = axis.normalize()
    #transfer_matrix = axis_view_matrix(axis)

    #print('t m:',transfer_matrix)

    new_point_list = []
    for point in point_list:
        new_point_list.append(transfer_matrix * point)

    return new_point_list
    

# define the axis 
# calculate the center and transfer matrix
axis = glm.vec3(1,1,1)
transfer_matrix = axis_view_matrix(axis=axis)
center = transfer_matrix * glm.vec3(32,32,32)

print('transfer_matrix: ',  str(transfer_matrix))

for program_length in data_label_paths:

    expressions = gen.programs[program_length]  
    for index,exp in enumerate(expressions):
        program = gen.parse(exp)

        sim.get_all_primitives(gen.primitives)
        sim.generate_stack(program, if_primitives=True)
        voxel = sim.stack.get_items()[0]

        #point_list = border_find_points(voxel)
        #center = glm.vec3(32,32,32)

        point_list = axis_view_place_points(voxel, transfer_matrix = transfer_matrix)

        #projection 
        #img = z_parrallel_projection(voxel, 32 , 32)
        img = z_parrallel_projection_point(point_list,origin_w=128,origin_h=128, w=128, h=128, center_x=center[0], center_y=center[1])

        img_mask = img * 255
        img_mask = np.array(img_mask,dtype=int)
        #img_mask = cv2.merge(img_mask)
        cv2.imwrite('data/2D/' + str(program_length) + '/' + str(index) +'.jpg' , img_mask)
            
        print('finish processing pic '+str(index))

    print('Finish processing ' + str(program_length) + ' instructions programs')



    
    

