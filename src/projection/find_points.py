import numpy as np
from src.display.glm import glm


def find_points(a):
    l = []
    for i in range(64):
        for j in range(64):
            for k in range(64):
                if a[i,j,k]== True:
                    l.append(glm.vec3(i,j,k))

    return l

def find_points_simple(a):
    l = []

    indexl = np.argwhere(a == True)

    for i in indexl:
        l.append(glm.vec3(i[0],i[1],i[2]))

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

def border_find_points_simple(a):
    zero_i = np.zeros([1,64,64] , dtype=bool)
    ai_minus_1 =  np.concatenate((zero_i,a[:-1,:,:]), axis = 0)
    ai_plus_1 = np.concatenate((a[1:,:,:] , zero_i), axis = 0)

    zero_j = np.zeros([64,1,64], dtype = bool)
    aj_minus_1 = np.concatenate((zero_j,a[:,:-1,:]), axis = 1)
    aj_plus_1 = np.concatenate((a[:,1:,:] , zero_j), axis = 1)

    zero_k = np.zeros([64,64,1], dtype = bool)
    ak_minus_1 = np.concatenate((zero_k,a[:,:,:-1]), axis = 2)
    ak_plus_1 = np.concatenate((a[:,:,1:] , zero_k), axis = 2)

    mat = np.logical_and(ai_minus_1, ai_plus_1)
    mat = np.logical_and(mat, aj_minus_1)
    mat = np.logical_and(mat, aj_plus_1)
    mat = np.logical_and(mat, ak_minus_1)
    mat = np.logical_and(mat, ak_plus_1)

    a_trim = (a * 1. - np.logical_and(a, mat) * 1.).astype(np.bool)

    indexl = np.argwhere(a_trim == True)

    #for i in indexl:
    #    l.append(glm.vec3(i[0],i[1],i[2]))

    return indexl

def border_find_planes(a):
    zero_i = np.zeros([1,64,64] , dtype=bool)
    ai_minus_1 =  np.concatenate((zero_i,a[:-1,:,:]), axis = 0)
    ai_plus_1 = np.concatenate((a[1:,:,:] , zero_i), axis = 0)

    zero_j = np.zeros([64,1,64], dtype = bool)
    aj_minus_1 = np.concatenate((zero_j,a[:,:-1,:]), axis = 1)
    aj_plus_1 = np.concatenate((a[:,1:,:] , zero_j), axis = 1)

    zero_k = np.zeros([64,64,1], dtype = bool)
    ak_minus_1 = np.concatenate((zero_k,a[:,:,:-1]), axis = 2)
    ak_plus_1 = np.concatenate((a[:,:,1:] , zero_k), axis = 2)

    x_positive = (a * 1. - np.logical_and(a, ai_plus_1) * 1.).astype(np.bool)
    x_negative = (a * 1. - np.logical_and(a, ai_minus_1) * 1.).astype(np.bool)

    y_positive = (a * 1. - np.logical_and(a, aj_plus_1) * 1.).astype(np.bool)
    y_negative = (a * 1. - np.logical_and(a, aj_minus_1) * 1.).astype(np.bool)

    z_positive = (a * 1. - np.logical_and(a, ak_plus_1) * 1.).astype(np.bool)
    z_negative = (a * 1. - np.logical_and(a, ak_minus_1) * 1.).astype(np.bool)

    index_x_positive = np.argwhere(x_positive == True)
    index_x_negative = np.argwhere(x_negative == True)

    index_y_positive = np.argwhere(y_positive == True)
    index_y_negative = np.argwhere(y_negative == True)

    index_z_positive = np.argwhere(z_positive == True)
    index_z_negative = np.argwhere(z_negative == True)

    return [index_x_positive, index_x_negative, index_y_positive, index_y_negative, index_z_positive, index_z_negative]