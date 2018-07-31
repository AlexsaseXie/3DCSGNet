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
    l = []

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

    for i in indexl:
        l.append(glm.vec3(i[0],i[1],i[2]))

    return l