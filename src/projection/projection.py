import math
import numpy as np
from src.display.glm import glm
from src.projection.find_points import border_find_points

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