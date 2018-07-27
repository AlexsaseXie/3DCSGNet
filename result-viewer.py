#encoding=utf8
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import time

from src.display.glm import glm
from src.display import transform
from src.projection.find_points import *

import random

eye = glm.vec3(0,0,0)
up = glm.vec3(0,1,0)
center = glm.vec3(32,32,32)


current_index = 0
is_target = False

file_path = 'trained_models/results/given-model.pth'

if (len(sys.argv) > 1):
    file_path = sys.argv[1]
with open(file_path + '/pred.txt') as data_file:
    expressions = data_file.readlines()
with open(file_path + '/target.txt') as target_data_file:
    target_expressions = target_data_file.readlines()


import deepdish as dd
from src.Utils.train_utils import voxels_from_expressions


# pre-rendered shape primitives in the form of voxels for better performance
primitives = dd.io.load("data/primitives.h5")
img_points = []
target_img_points = []

#info text
print('HELP TEXT:')
print('\n ----------------------------- \n')
print ('wasd : rotate the view point')
print ('nm : drag the view point nearer or further')
print ('r : reset eye and view direction')
print ('t : toggle to view predicted model and target model')
print ('Left and Right Key : view next model\'s voxel representation')
print('\n ----------------------------- \n')



def init_img():
    global current_index, img_points, target_img_points
    print('loading models...')

    expression = expressions[current_index]
    target_expression = target_expressions[current_index]

    voxel = voxels_from_expressions([expression, target_expression], primitives, max_len=7)

    img_points = border_find_points(voxel[0])
    target_img_points = border_find_points(voxel[1])

    print('loading fish!')

def single_plane(a,b,c,d):
    glBegin(GL_QUADS)

    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(a.x, a.y, a.z)
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(b.x, b.y, b.z)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(c.x, c.y, c.z)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(d.x, d.y, d.z)

    glEnd()

def single_cube(center, l):
    glBegin(GL_QUADS)

    #front
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] + l/2)
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] + l/2)
    if not is_target:
        glColor3f(0.0, 0.3, 0.3)
    else:
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] - l/2)
    if not is_target:
        glColor3f(0.0, 0.3, 0.3)
    else:
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] - l/2)

    #right
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] + l/2)
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] + l/2)
    if not is_target:         
        glColor3f(0.0, 0.3, 0.3)     
    else:         
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] - l/2)
    if not is_target:         
        glColor3f(0.0, 0.3, 0.3)     
    else:         
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] - l/2)

    #back
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] + l/2)
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] + l/2)
    if not is_target:         
        glColor3f(0.0, 0.3, 0.3)     
    else:         
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] - l/2)
    if not is_target:         
        glColor3f(0.0, 0.3, 0.3)     
    else:         
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] - l/2)

    #left
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] + l/2)
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] + l/2)
    if not is_target:         
        glColor3f(0.0, 0.3, 0.3)     
    else:         
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] - l/2)
    if not is_target:         
        glColor3f(0.0, 0.3, 0.3)     
    else:         
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] - l/2)

    #up 
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] + l/2)
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] + l/2)
    if not is_target:         
        glColor3f(0.0, 0.3, 0.3)     
    else:         
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] + l/2)
    if not is_target:         
        glColor3f(0.0, 0.3, 0.3)     
    else:         
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] + l/2)

    #down 
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] - l/2)
    if not is_target:
        glColor3f(0.3, 0.3, 0.0)
    else:
        glColor3f(1.0, 0.0, 1.0)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] - l/2)
    if not is_target:         
        glColor3f(0.0, 0.3, 0.3)     
    else:         
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] - l/2)
    if not is_target:         
        glColor3f(0.0, 0.3, 0.3)     
    else:         
        glColor3f(1.0, 1.0, 0.0)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] - l/2)

    glEnd()

def display():
    global xaxis,yaxis,zaxis
    time.sleep(0.1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glLoadIdentity()

    gluLookAt(eye.x, eye.y, eye.z , center.x , center.y , center.z, up.x, up.y, up.z)
    #cubes()

    if not is_target:
        for p in img_points:
            single_cube(p,1)
    else:
        for p in target_img_points:
            single_cube(p,1)
    
    glutSwapBuffers()

def reshape(w, h):
    if (h == 0):
        h = 1
    glViewport(0, 0, w,h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0,w /h, 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)

def init(width, height):
    if (height == 0):
        height = 1
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)

    #glLightfv(GL_LIGHT0, GL_POSITION, 1.0, 1.0, 1.0, 0.0)
    #glLightfv(GL_LIGHT0, GL_DIFFUSE, 1.0, 1.0, 1.0, 1.0)
    #glLightfv(GL_LIGHT0, GL_SPECULAR, 1.0, 1.0, 1.0, 1.0)
    #glLightModelfv(GL_LIGHT_MODEL_AMBIENT, 0.2, 0.2, 0.2, 1.0)

    #glEnable(GL_LIGHTING)
    #glEnable(GL_LIGHT0)

    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0,width /height, 1, 500.0)
    glMatrixMode(GL_MODELVIEW)

    init_img()
    printExpression()
    showParameters()

def keyboard(key, w, h):
    global eye, up, center, is_target
    if (key==b'a' or key == b'A'):
        eye,up = transform.left(5.0, eye, up, center)
    if (key==b'd' or key == b'D'):
        eye,up = transform.left(-5.0, eye, up, center)
    if (key==b'w' or key == b'W'):
        eye,up = transform.up(5.0, eye, up, center)
    if (key==b's' or key ==b'S'):
        eye,up = transform.up(-5.0, eye, up, center)


    if (key==b'n' or key == b'N'):
        eye,up = transform.near(2, eye, up, center)
    if (key==b'm' or key == b'M'):
        eye,up = transform.near(-2, eye, up, center)


    if (key==b'r' or key == b'R'):
        reset()
    if (key==b't' or key == b'T'):
        is_target = ~is_target
    

    showParameters()
    glutPostRedisplay()

def specialKeyboard(key, w, h):
    global current_index, expressions
    if (key == GLUT_KEY_LEFT):
        sys.stdout.write('\n')
        print('Left pressed')
        current_index -= 1
        current_index = current_index % len(expressions)
        init_img()

        reset()
        printExpression()

    if (key == GLUT_KEY_RIGHT):
        sys.stdout.write('\n')
        print('Right pressed')
        current_index += 1
        current_index = current_index % len(expressions)
        init_img()

        reset()
        printExpression()
    
    if (key == GLUT_KEY_DOWN):
        sys.stdout.write('\n')
        print('Random Model')
        current_index = random.randint(0, len(target_expressions) - 1)
        init_img()

        reset()
        printExpression()

    showParameters()
    glutPostRedisplay()

def reset():
    global eye, up, center, is_target

    eye.x = 0
    eye.y = 0 
    eye.z = 0

    up.x = 0
    up.y = 1
    up.z = 0

    center.x = 32
    center.y = 32
    center.z = 32

    is_target = False

def printExpression():
    print('case ' + str(current_index) + ':')
    print('predict: ' + expressions[current_index] )
    print('target: ' + target_expressions[current_index])

previous_len = 0
def showParameters():
    global previous_len

    sys.stdout.write(' ' * (previous_len + 1) + '\r')
    sys.stdout.flush()
    output_str = 'pic '+ str(current_index) +' | eye:' + str(eye) + '; center:' + str(center) + '; up:' + str(up) + '\r'
    sys.stdout.write(output_str)
    sys.stdout.flush()

    previous_len = len(output_str)
    
glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowPosition(400, 100)
glutInitWindowSize(640, 480)
glutCreateWindow("CSG test result viewer")
glutDisplayFunc(display)
#glutIdleFunc(display)
glutReshapeFunc(reshape)
glutKeyboardFunc(keyboard)
glutSpecialFunc(specialKeyboard)
init(640, 480)
glutMainLoop()
