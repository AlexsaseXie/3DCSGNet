import numpy as np
import cv2
import math



from src.Generator.generator import *
from src.Generator.stack import *

data_label_paths = {3: "../data/one_op/expressions.txt" }
            #5: "../data/two_ops/expressions.txt",
            #7: "../data/three_ops/expressions.txt"}

gen = Generator(data_labels_paths=data_label_paths, primitives={})

sim = SimulateStack(max_len=5, canvas_shape=[64,64,64], draw_uniques=None)


def find_points(a):
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

def z_parrallel_projection(point_list ,w ,h ):
    img = np.zeros([w,h],dtype=float)

    w_ratio = w / 64.0
    h_ratio = h / 64.0

    for i in range(w):
        for j in range(h):
            for point in point_list:
                if abs(i / w_ratio - point[0]) <= 1 and abs(j / h_ratio - point[1] <= 1):
                    img[i,j] += 1.0 / 8.0 
    
    return img
    
    

with open(data_label_paths[3]) as data_file:
    expressions = data_file.readlines()
    
for exp in expressions:
    program = gen.parse(exp)

    sim.generate_stack(program, if_primitives=False)
    voxel = sim.stack.get_items()[0]

    point_list = find_points(voxel)

    #projection 
    img = z_parrallel_projection(point_list, 64 , 64)

    img_mask = np.array(img) * 255
    img_mask = cv2.merge(img_mask)
    cv2.imwrite('test.jpg' , img_mask)

    break

        



    
    

