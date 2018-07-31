import math
import numpy as np
from src.display.glm import glm
from src.projection.find_points import *

import time

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
def z_parrallel_projection_point(point_list, center_x , center_y , origin_w=128, origin_h=128 , origin_z=128 , w=128, h=128 ):
    img = np.ones([w,h],dtype=float)
    depth = np.zeros([w,h], dtype=float)

    point_zs = np.zeros([origin_w,origin_h], dtype=float) 

    tick = time.time()

    # clean and then index the points 
    # may have negative index on p_x or p_y, but this doesn't matter. We refer to those elements by the same
    # negative index
    for point in point_list:
        p_x = math.floor(point[0])
        p_y = math.floor(point[1])
        p_z = point[2]

        p_x_ceiling = math.ceil(point[0])
        p_y_ceiling = math.ceil(point[1])

        point_zs[p_x,p_y] = max(point_zs[p_x,p_y], p_z)
        if (p_x_ceiling != p_x):
            point_zs[p_x_ceiling,p_y] = max(point_zs[p_x_ceiling,p_y], p_z)
        if (p_y_ceiling != p_y):
            point_zs[p_x,p_y_ceiling] = max(point_zs[p_x,p_y_ceiling], p_z)
        if (p_x_ceiling != p_x and p_y_ceiling != p_y):
            point_zs[p_x_ceiling,p_y_ceiling] = max(point_zs[p_x_ceiling,p_y_ceiling], p_z)

    print('summarize point_list cost:' + str(time.time()-tick) + 'sec')
    tick = time.time()


    w_ratio = origin_w / w
    h_ratio = origin_h / h


    # (center_x, center_y) -> (w/2, h/2)
    for i in range(w):
        for j in range(h):
            # candidate area
            x_min = math.floor(i - w/2 + 0.001) * w_ratio + center_x
            x_max = math.ceil(i - w/2 + 0.001) * w_ratio + center_x
            y_min = math.floor(j - h/2 + 0.001) * h_ratio + center_y
            y_max = math.ceil(j - h/2 + 0.001) * h_ratio + center_y

            depth = 0

            x = math.ceil(x_min)
            while( x < x_max ):
                y = math.ceil(y_min)
                while( y < y_max ):
                    point_z = point_zs[x,y]

                    if point_z > depth:
                        depth = point_z
                            
                        img[i][j] = 1 - depth * 1.0 / origin_z
                    
                    y = y + 1

                x = x + 1

    print('Do w*h projection cost:' + str(time.time() - tick) + 'sec')

    return img

# origin_w : x side of the view
# origin_h : y side of the view
def z_parrallel_projection_point_simple(point_list, center_x , center_y , origin_w=128, origin_h=128 , origin_z=128 , w=128, h=128 ):
    #tick = time.time()
    img = np.ones([w,h],dtype=float)
    depth = np.zeros([w,h], dtype=float)

    w_ratio = origin_w / w
    h_ratio = origin_h / h



    # clean and then index the points 
    # may have negative index on p_x or p_y, but this doesn't matter. We refer to those elements by the same
    # negative index
    for point in point_list:
        p_x = point[0]
        p_y = point[1]
        p_z = point[2]

        # candidate area
        x_min = math.floor(p_x - center_x + 0.001) / w_ratio + w/2
        x_max = math.ceil(p_x - center_x + 0.001) / w_ratio + w/2
        y_min = math.floor(p_y - center_y + 0.001) / h_ratio + h/2
        y_max = math.ceil(p_y - center_y + 0.001) / h_ratio + h/2

        # x = math.ceil(x_min)
        # while( x <= x_max ):
        #     y = math.ceil(y_min)
        #     while( y <= y_max ):
        #         if p_z > depth[x,y]:
        #             depth[x][y] = p_z
                        
        #             img[x][y] = 1 - depth[x][y] * 1.0 / origin_z
                
        #         y = y + 1

        #     x = x + 1

        # !!! special for target 128 size picture
        x = math.ceil(x_min)
        y = math.ceil(y_min)
        if p_z > depth[x,y]:
            depth[x][y] = p_z
                
            img[x][y] = 1 - depth[x][y] * 1.0 / origin_z

        if x + 1 <= x_max and p_z > depth[x+1,y]:
            depth[x+1][y] = p_z
                
            img[x+1][y] = 1 - depth[x+1][y] * 1.0 / origin_z

        if y + 1 <= y_max and p_z > depth[x,y+1]:
            depth[x][y+1] = p_z
                
            img[x][y+1] = 1 - depth[x][y+1] * 1.0 / origin_z

        if x + 1 <= x_max and y + 1 <= y_max and p_z > depth[x+1,y+1]:
            depth[x+1][y+1] = p_z
                
            img[x+1][y+1] = 1 - depth[x+1][y+1] * 1.0 / origin_z
            
        

    #print('summarize point_list cost:' + str(time.time()-tick) + 'sec')
    return img


# return the transfer matrix of axis rotation
def axis_view_matrix(axis: glm.vec3):
    zaxis = axis.normalize()

    yaxis = glm.vec3(1,1,(- zaxis.x - zaxis.y )/zaxis.z)
    yaxis = yaxis.normalize()

    xaxis = yaxis.cross(zaxis)
    xaxis = xaxis.normalize()

    # return glm.mat3(xaxis.x,xaxis.y,xaxis.z,
    #     yaxis.x,yaxis.y,yaxis.z,
    #     zaxis.x,zaxis.y,zaxis.z)

    return np.array([[xaxis.x,xaxis.y,xaxis.z], [yaxis.x,yaxis.y,yaxis.z], [zaxis.x,zaxis.y,zaxis.z]],dtype=float)

# see from the transfered top toward -z' axis
# axis : the new z axis under the previous coordinate
def axis_view_place_points(voxel ,transfer_matrix):
    tick = time.time()

    point_list = border_find_points_simple(voxel)

    print(point_list.shape)
    print('find points cost:' + str(time.time()-tick) + ' sec')

    tick = time.time()

    new_point_list = []
    for point in point_list:
        # new_point_list.append(transfer_matrix * point)
        new_point_list.append(np.dot(transfer_matrix,point))

    print('multiply cost:' + str(time.time()-tick) + ' sec')

    return new_point_list