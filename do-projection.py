import numpy as np
import cv2
import math
import time

from src.display.glm import glm
from src.Generator.generator import *
from src.Generator.stack import *

from src.projection.find_points import *
from src.projection.projection import *

data_label_paths = {3: "data/one_op/expressions.txt" }
            #5: "/data/two_ops/expressions.txt",
            #7: "/data/three_ops/expressions.txt"}

gen = Generator(data_labels_paths=data_label_paths, primitives=None)

sim = SimulateStack(max_len=5, canvas_shape=[64,64,64], draw_uniques=None)
sim.get_all_primitives(gen.primitives)

# define the axis 
# calculate the center and transfer matrix
axis = glm.vec3(1,1,1)
transfer_matrix = axis_view_matrix(axis=axis)
center = np.dot( transfer_matrix,np.array([32,32,32],dtype=float) )

print('transfer_matrix: ',  str(transfer_matrix))

for program_length in data_label_paths:

    expressions = gen.programs[program_length]

    tick = time.time()
    for index,exp in enumerate(expressions):
        program = gen.parse(exp)

        sim.generate_stack(program, if_primitives=True)
        voxel = sim.stack.items[0]

        #point_list = border_find_points(voxel)
        #center = glm.vec3(32,32,32)

        #tick = time.time()

        point_list = axis_view_place_points(voxel, transfer_matrix = transfer_matrix)

        #print('generate point list in ' + str(time.time()-tick) + ' sec')

        #projection 
        #img = z_parrallel_projection(voxel, 32 , 32)
        img = z_parrallel_projection_point_simple(point_list,origin_w=128,origin_h=128, origin_z=128, w=128, h=128, center_x=center[0], center_y=center[1])

        img_mask = img * 255
        img_mask = np.array(img_mask,dtype=int)
        
        cv2.imwrite('data/2D-depth-simple/' + str(program_length) + '/' + str(index) +'.jpg' , img_mask)

        if index % 200 == 0:    
            print('finish processing pic '+str(index) + ' in ' + str(time.time()-tick) + ' sec\n')

            tick = time.time()

    print('Finish processing ' + str(program_length) + ' instructions programs')



    
    

