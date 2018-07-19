#encoding=utf8
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import time

from glm import glm
import transform

eye = glm.vec3(0,0,5)
up = glm.vec3(0,1,0)

xaxis = 0.0
yaxis = 0.0
zaxis = 1.0


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
    gluLookAt(eye.x, eye.y, eye.z ,0,0,0, up.x, up.y, up.z)
    cubes()
    #xaxis = xaxis + 1
    #yaxis = yaxis + 1
    #zaxis = zaxis + 1
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
        eye,up = transform.left(5.0, eye, up)
    if (key==b'd' or key == b'D'):
        print('d Pressed')
        eye,up = transform.left(-5.0, eye, up)
    if (key==b'w' or key == b'W'):
        print('w Pressed')
        eye,up = transform.up(5.0, eye, up)
    if (key==b's' or key ==b'S'):
        print('s Pressed')
        eye,up = transform.up(-5.0, eye, up)


    if (key==b'n' or key == b'N'):
        print('Near')
        eye,up = transform.near(0.5, eye, up)
    if (key==b'm' or key == b'M'):
        print('Further')
        eye,up = transform.near(-0.5, eye, up)

    glutPostRedisplay()

print ('X：X轴+10 Y：Y轴+10 Z：Z轴+10')
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
