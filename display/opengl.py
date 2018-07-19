#encoding=utf8
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import time

xaxis = 0.0
yaxis = 0.0
zaxis = 0.0


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
    glTranslatef(0, 0, -5)
    #glRotatef(xaxis, 1, 0, 0)
    #glRotatef(yaxis, 0, 1, 0)
    #glRotatef(zaxis, 0, 0, 1)
    gluLookAt(xaxis,yaxis,zaxis,0,0,0,0,1,0)
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
    if (key==b'x' or key == b'X'):
        print('x Pressed')
        xaxis = xaxis + 10
    if (key==b'y' or key == b'Y'):
        print('y Pressed')
        yaxis = yaxis + 10
    if (key==b'z' or key == b'Z'):
        print('z Pressed')
        zaxis = zaxis + 10

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
