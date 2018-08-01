from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtOpenGL import QGLWidget
import sys
from OpenGL.GL import *
from OpenGL.GLU import *

from src.display.glm import glm
from src.display import transform
from src.projection.find_points import *

import deepdish as dd
from src.Utils.train_utils import voxels_from_expressions
import random


class GlobalStorage:
    def __init__(self):
        self.eye = glm.vec3(0,0,0)
        self.up = glm.vec3(0,1,0)
        self.center = glm.vec3(32,32,32)
        self.current_index = 0
        self.is_target = False
        self.file_path = 'trained_models/results/given-model.pth'

        with open(self.file_path + '/beam_10_pred.txt') as data_file:
            self.expressions = data_file.readlines()
        with open(self.file_path + '/beam_10_target.txt') as target_data_file:
            self.target_expressions = target_data_file.readlines()

        self.primitives = dd.io.load("data/primitives.h5")
        self.img_points = []
        self.target_img_points = []

 
class MainWindow(QMainWindow):
    """docstring for Mainwindow"""
    def __init__(self, storage = None, parent = None):
        super(MainWindow,self).__init__(parent)
        self.storage = storage
        
        self.basic()
        splitter_main = self.split_()
        self.setCentralWidget(splitter_main)
 
	#窗口基础属性
    def basic(self):
        #设置标题，大小，图标
        self.setWindowTitle("GT")
        self.resize(1100,650)
        self.setWindowIcon(QIcon("./image/Gt.png"))
        #居中显示
        screen = QDesktopWidget().geometry()
        self_size = self.geometry()
        self.move((screen.width() - self_size.width())/2,(screen.height() - self_size.height())/2)
 
	#分割窗口
    def split_(self):
        self.s = OpenGLWidget(storage = self.storage)   #将opengl例子嵌入GUI
        splitter_main = QSplitter(Qt.Horizontal)
        self.textedit_main = QTextEdit()
        splitter_main.addWidget(self.textedit_main)
        splitter_main.addWidget(self.s)
        splitter_main.setStretchFactor(0,2)
        splitter_main.setStretchFactor(2,4)
        return splitter_main
    
    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()

class LeftWidget(QWidget):
    def __init__(self, storage = None):
        super(LeftWidget, self,).__init__()        
        self.storage = storage

class OpenGLWidget(QGLWidget):
    def __init__(self, storage = None):
        super(OpenGLWidget,self).__init__()
        self.storage = storage

    def init_img(self):
        print('loading models...')

        expression = self.storage.expressions[self.storage.current_index]
        target_expression = self.storage.target_expressions[self.storage.current_index]

        voxel = voxels_from_expressions([expression, target_expression], self.storage.primitives, max_len=7)

        self.storage.img_points = border_find_points_simple(voxel[0])
        self.storage.target_img_points = border_find_points_simple(voxel[1])

        print('loading fish!')

    def reset(self):
        self.storage.eye.x = 0
        self.storage.eye.y = 0 
        self.storage.eye.z = 0

        self.storage.up.x = 0
        self.storage.up.y = 1
        self.storage.up.z = 0

        self.storage.center.x = 32
        self.storage.center.y = 32
        self.storage.center.z = 32

        self.storage.is_target = False

    def single_cube(self, center, l):
        glBegin(GL_QUADS)

        #front
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] + l/2, center[1] - l/2, center[2] + l/2)
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] + l/2, center[1] + l/2, center[2] + l/2)
        if not self.storage.is_target:
            glColor3f(0.0, 0.3, 0.3)
        else:
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] + l/2, center[1] + l/2, center[2] - l/2)
        if not self.storage.is_target:
            glColor3f(0.0, 0.3, 0.3)
        else:
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] + l/2, center[1] - l/2, center[2] - l/2)

        #right
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] + l/2, center[1] + l/2, center[2] + l/2)
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] - l/2, center[1] + l/2, center[2] + l/2)
        if not self.storage.is_target:         
            glColor3f(0.0, 0.3, 0.3)     
        else:         
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] - l/2, center[1] + l/2, center[2] - l/2)
        if not self.storage.is_target:         
            glColor3f(0.0, 0.3, 0.3)     
        else:         
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] + l/2, center[1] + l/2, center[2] - l/2)

        #back
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] - l/2, center[1] - l/2, center[2] + l/2)
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] - l/2, center[1] + l/2, center[2] + l/2)
        if not self.storage.is_target:         
            glColor3f(0.0, 0.3, 0.3)     
        else:         
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] - l/2, center[1] + l/2, center[2] - l/2)
        if not self.storage.is_target:         
            glColor3f(0.0, 0.3, 0.3)     
        else:         
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] - l/2, center[1] - l/2, center[2] - l/2)

        #left
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] + l/2, center[1] - l/2, center[2] + l/2)
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] - l/2, center[1] - l/2, center[2] + l/2)
        if not self.storage.is_target:         
            glColor3f(0.0, 0.3, 0.3)     
        else:         
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] - l/2, center[1] - l/2, center[2] - l/2)
        if not self.storage.is_target:         
            glColor3f(0.0, 0.3, 0.3)     
        else:         
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] + l/2, center[1] - l/2, center[2] - l/2)

        #up 
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] + l/2, center[1] - l/2, center[2] + l/2)
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] - l/2, center[1] - l/2, center[2] + l/2)
        if not self.storage.is_target:         
            glColor3f(0.0, 0.3, 0.3)     
        else:         
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] - l/2, center[1] + l/2, center[2] + l/2)
        if not self.storage.is_target:         
            glColor3f(0.0, 0.3, 0.3)     
        else:         
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] + l/2, center[1] + l/2, center[2] + l/2)

        #down 
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] + l/2, center[1] - l/2, center[2] - l/2)
        if not self.storage.is_target:
            glColor3f(0.3, 0.3, 0.0)
        else:
            glColor3f(1.0, 0.0, 1.0)
        glVertex3f(center[0] - l/2, center[1] - l/2, center[2] - l/2)
        if not self.storage.is_target:         
            glColor3f(0.0, 0.3, 0.3)     
        else:         
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] - l/2, center[1] + l/2, center[2] - l/2)
        if not self.storage.is_target:         
            glColor3f(0.0, 0.3, 0.3)     
        else:         
            glColor3f(1.0, 1.0, 0.0)
        glVertex3f(center[0] + l/2, center[1] + l/2, center[2] - l/2)

        glEnd()

    def initializeGL(self):
        #self.init_global_params()

        # otherwise right widget won't be focused on
        self.setFocusPolicy(Qt.StrongFocus)

        # glClearColor(1,0,0,1)
        # glEnable(GL_DEPTH_TEST)
        # glEnable(GL_LIGHT0)
        # glEnable(GL_LIGHTING)
        # glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        # glEnable(GL_COLOR_MATERIAL)
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)

        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, 640 / 480, 1, 500.0)
        glMatrixMode(GL_MODELVIEW)

        self.init_img()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glLoadIdentity()

        gluLookAt(self.storage.eye.x, self.storage.eye.y, self.storage.eye.z , 
                self.storage.center.x , self.storage.center.y , self.storage.center.z, 
                self.storage.up.x, self.storage.up.y, self.storage.up.z)

        if not self.storage.is_target:
            for p in self.storage.img_points:
                self.single_cube(p,1)
        else:
            for p in self.storage.target_img_points:
                self.single_cube(p,1)

    def resizeGL(self, w , h ):
        glViewport(0, 0, w,h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0,w /h, 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)

    def keyPressEvent(self, e):
        # control event

        if e.key() == QtCore.Qt.Key_A:
            self.storage.eye, self.storage.up = transform.left(5.0, self.storage.eye, self.storage.up, self.storage.center)
            self.updateGL()
        if e.key() == QtCore.Qt.Key_D:
            self.storage.eye, self.storage.up = transform.left(-5.0, self.storage.eye, self.storage.up, self.storage.center)
            self.updateGL()
        if e.key() == QtCore.Qt.Key_W:
            self.storage.eye, self.storage.up = transform.up(5.0, self.storage.eye, self.storage.up, self.storage.center)
            self.updateGL()
        if e.key() == QtCore.Qt.Key_S:
            self.storage.eye, self.storage.up = transform.up(-5.0, self.storage.eye, self.storage.up, self.storage.center)
            self.updateGL()


        if e.key() == QtCore.Qt.Key_N:
            self.storage.eye, self.storage.up = transform.near(2, self.storage.eye, self.storage.up, self.storage.center)
            self.updateGL()
        if e.key() == QtCore.Qt.Key_M:
            self.storage.eye, self.storage.up = transform.near(-2, self.storage.eye, self.storage.up, self.storage.center)
            self.updateGL()


        if e.key() == QtCore.Qt.Key_R:
            self.reset()
            self.updateGL()
        if e.key() == QtCore.Qt.Key_T:
            self.storage.is_target = ~ self.storage.is_target
            self.updateGL()


        if e.key() == QtCore.Qt.Key_Left:
            self.storage.current_index -= 1
            self.storage.current_index = self.storage.current_index % len(self.storage.expressions)
            self.init_img()

            self.reset()
            self.updateGL()

        if e.key() == QtCore.Qt.Key_Right:
            self.storage.current_index += 1
            self.storage.current_index = self.storage.current_index % len(self.storage.expressions)
            self.init_img()

            self.reset()
            self.updateGL()
        
        if e.key() == QtCore.Qt.Key_Down:
            self.storage.current_index = random.randint(0, len(self.storage.expressions) - 1)
            self.init_img()

            self.reset()
            self.updateGL()
        
        
 
if __name__ == "__main__":
    app = QApplication(sys.argv)

    #storage
    storage = GlobalStorage()

    win = MainWindow(storage=storage)
    win.show()
    sys.exit(app.exec_())