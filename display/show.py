from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


eye = [0.0, 0.0, -5.0]
up = [0.0, 1.0, 0.0]

def drawFunc():
    glClear(GL_COLOR_BUFFER_BIT)

    gluLookAt( eye[0],eye[1],eye[2] , 0,0,0, up[0], up[1], up[2])
    glutWireTeapot(0.5)
    glFlush()

def keyFunc(key, x, y):
    global eye, up
    if (key == 'a') :
        eye = [1.0, 0, 0]
    #elif (key == 'r') :
    #    eye = [0, 0, 1.0]
    glutPostRedisplay()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
    glutInitWindowSize(400,400)
    glutCreateWindow("ShowPic")
    glutDisplayFunc(drawFunc)
    # glutIdleFunc(drawFunc)
    glutKeyboardFunc(keyFunc)
    glutMainLoop()

main()
