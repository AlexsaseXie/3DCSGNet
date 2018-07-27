from src.display.glm import glm


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