#encoding=utf8
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import time

from glm import glm
import transform

eye = glm.vec3(0,0,0)
up = glm.vec3(0,1,0)
center = glm.vec3(32,32,32)


current_index = 0

file_path = sys.argv[1]
with open(file_path) as data_file:
    expressions = data_file.readlines()

import deepdish as dd
sys.path.append('..')
from src.Utils.train_utils import voxels_from_expressions

# pre-rendered shape primitives in the form of voxels for better performance
primitives = dd.io.load("data/primitives.h5")

expression = expressions[current_index]
voxel = voxels_from_expressions([expression], primitives, max_len=7)

def fake_data():
    a = np.zeros([64,64,64],dtype=bool)

    l = []
    for i in range(100):
        temp = np.random.randint(0,64,size=3)
        l.append(temp)
        a[temp[0],temp[1],temp[2]] = True
    return a, l

def fake_cube():
    a = np.zeros([64,64,64],dtype=bool)

    l = []
    for i in range(22,43):
        for j in range(22,43):
            for k in range(22,43):
                a[i,j,k] = True

    return a,l

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

img,_ = fake_cube()
img_points = find_points(img)

print(len(img_points))

def cubes():
    glBegin(GL_QUADS)
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(0.3, 0.3, -0.3)
    glColor3f(0.0, 0.3, 0.0)
    glVertex3f(-0.3, 0.3, -0.3)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(-0.3, 0.3, 0.3)
    glColor3f(0.3, 0.3, 0.3)
    glVertex3f(0.3, 0.3, 0.3)

    glColor3f(0.3, 0.3, 0.3)
    glVertex3f(0.3, 0.3, 0.3)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(-0.3, 0.3, 0.3)
    glColor3f(0.0, 0.0, 0.3)
    glVertex3f(-0.3, -0.3, 0.3)
    glColor3f(0.3, 0.0, 0.3)
    glVertex3f(0.3, -0.3, 0.3)

    glColor3f(0.3, 0.0, 0.0)
    glVertex3f(0.3, -0.3, -0.3)
    glColor3f(0.0, 0.0, 0.0)
    glVertex3f(-0.3, -0.3, -0.3)
    glColor3f(0.0, 0.3, 0.0)
    glVertex3f(-0.3, 0.3, -0.3)
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(0.3, 0.3, -0.3)

    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(-0.3, 0.3, 0.3)
    glColor3f(0.0, 0.3, 0.0)
    glVertex3f(-0.3, 0.3, -0.3)
    glColor3f(0.0, 0.0, 0.0)
    glVertex3f(-0.3, -0.3, -0.3)
    glColor3f(0.0, 0.0, 0.3)
    glVertex3f(-0.3, -0.3, 0.3)

    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(0.3, 0.3, -0.3)
    glColor3f(0.3, 0.3, 0.3)
    glVertex3f(0.3, 0.3, 0.3)
    glColor3f(0.3, 0.0, 0.3)
    glVertex3f(0.3, -0.3, 0.3)
    glColor3f(0.3, 0.0, 0.0)
    glVertex3f(0.3, -0.3, -0.3)

    glColor3f(0.5, 0.5, 0.0)
    glVertex3f(0.5, 0.5, -0.5)
    glColor3f(0.0, 0.5, 0.0)
    glVertex3f(-0.5, 0.5, -0.5)
    glColor3f(0.0, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glColor3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)

    glColor3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glColor3f(0.0, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glColor3f(0.0, 0.0, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glColor3f(0.5, 0.0, 0.5)
    glVertex3f(0.5, -0.5, 0.5)

    glColor3f(0.5, 0.0, 0.0)
    glVertex3f(0.5, -0.5, -0.5)
    glColor3f(0.0, 0.0, 0.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glColor3f(0.0, 0.5, 0.0)
    glVertex3f(-0.5, 0.5, -0.5)
    glColor3f(0.5, 0.5, 0.0)
    glVertex3f(0.5, 0.5, -0.5)

    glColor3f(0.0, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glColor3f(0.0, 0.5, 0.0)
    glVertex3f(-0.5, 0.5, -0.5)
    glColor3f(0.0, 0.0, 0.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glColor3f(0.0, 0.0, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)

    glColor3f(0.5, 0.5, 0.0)
    glVertex3f(0.5, 0.5, -0.5)
    glColor3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glColor3f(0.5, 0.0, 0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glColor3f(0.5, 0.0, 0.0)
    glVertex3f(0.5, -0.5, -0.5)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, 1.0, -1.0)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glColor3f(0.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glColor3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)

    glColor3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glColor3f(0.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glColor3f(0.0, 0.0, 0.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, 1.0, -1.0)

    glColor3f(0.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glColor3f(0.0, 0.0, 0.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, 1.0, -1.0)
    glColor3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glEnd()

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
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] + l/2)
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] + l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] - l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] - l/2)

    #right
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] + l/2)
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] + l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] - l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] - l/2)

    #back
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] + l/2)
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] + l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] - l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] - l/2)

    #left
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] + l/2)
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] + l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] - l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] - l/2)

    #up 
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] + l/2)
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] + l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] + l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] + l/2)

    #down 
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] + l/2, center[1] - l/2, center[2] - l/2)
    glColor3f(0.3, 0.3, 0.0)
    glVertex3f(center[0] - l/2, center[1] - l/2, center[2] - l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] - l/2, center[1] + l/2, center[2] - l/2)
    glColor3f(0.0, 0.3, 0.3)
    glVertex3f(center[0] + l/2, center[1] + l/2, center[2] - l/2)

    glEnd()

def display():
    global xaxis,yaxis,zaxis
    time.sleep(0.1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(1.0,1.0,1.0,0.0)
    glLoadIdentity()
    #glTranslatef(0, 0, -5)
    #glRotatef(xaxis, 1, 0, 0)
    #glRotatef(yaxis, 0, 1, 0)
    #glRotatef(zaxis, 0, 0, 1)
    gluLookAt(eye.x, eye.y, eye.z , center.x , center.y , center.z, up.x, up.y, up.z)
    #cubes()

    for p in img_points:
        single_cube(p,1)
    
    glutSwapBuffers()


def reshape(w, h):
    if (h == 0):
        h = 1
    glViewport(0, 0, w,h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0,w /h, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def init(width, height):
    if (height == 0):
        height = 1
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0,width /height, 1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def keyboard(key, w, h):
    global xaxis,yaxis,zaxis
    global eye, up
    if (key==b'a' or key == b'A'):
        print('a Pressed')
        eye,up = transform.left(5.0, eye, up, center)
    if (key==b'd' or key == b'D'):
        print('d Pressed')
        eye,up = transform.left(-5.0, eye, up, center)
    if (key==b'w' or key == b'W'):
        print('w Pressed')
        eye,up = transform.up(5.0, eye, up, center)
    if (key==b's' or key ==b'S'):
        print('s Pressed')
        eye,up = transform.up(-5.0, eye, up, center)


    if (key==b'n' or key == b'N'):
        print('Near')
        eye,up = transform.near(0.5, eye, up, center)
    if (key==b'm' or key == b'M'):
        print('Further')
        eye,up = transform.near(-0.5, eye, up, center)

    glutPostRedisplay()

print ('wasd : rotate the view point')
print ('nm : drag the view point nearer or further')
glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowPosition(400, 100)
glutInitWindowSize(640, 480)
glutCreateWindow("HiddenStrawberry")
glutDisplayFunc(display)
#glutIdleFunc(display)
glutReshapeFunc(reshape)
glutKeyboardFunc(keyboard)
init(640, 480)
glutMainLoop()
