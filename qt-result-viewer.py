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

class Communicate(QObject):

    rightChangeIndex = pyqtSignal()
    updateParams = pyqtSignal()
 
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
        self.right = OpenGLWidget(storage = self.storage, main_window = self)   #将opengl例子嵌入GUI
        splitter_main = QSplitter(Qt.Horizontal)
        self.left = LeftWidget(storage = self.storage, main_window = self)
        splitter_main.addWidget(self.left)
        splitter_main.addWidget(self.right)
        splitter_main.setStretchFactor(0,2)
        splitter_main.setStretchFactor(2,4)
        return splitter_main
    
    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()

    ###SLOTS:
    def leftChangeIndex(self):
        self.storage.current_index = int(self.left.indexEdit.text())

        self.left.indexEdit.setText(str(self.storage.current_index))
        
        self.right.init_img()

        self.right.reset()
        self.right.updateGL()

    def rightChangeIndex(self):
        self.left.indexEdit.setText(str(self.storage.current_index))

        self.right.init_img()

        self.right.reset()
        self.right.updateGL()

    def updateParams(self):
        self.left.eyeEdit.setText(str(self.storage.eye))
        self.left.upEdit.setText(str(self.storage.up))
        self.left.centerEdit.setText(str(self.storage.center))

class LeftWidget(QWidget):
    def __init__(self, storage = None, main_window = None):
        super(LeftWidget, self,).__init__()        
        self.storage = storage
        self.main_window = main_window

        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        grid.setSpacing(10)

        self.eyeTitle = QLabel('eye')
        self.eyeEdit = QLabel(str(self.storage.eye))

        grid.addWidget(self.eyeTitle,1, 0)
        grid.addWidget(self.eyeEdit,1, 1)

        self.upTitle = QLabel('up')
        self.upEdit = QLabel(str(self.storage.up))

        grid.addWidget(self.upTitle, 2, 0)
        grid.addWidget(self.upEdit, 2, 1)

        self.centerTitle = QLabel('center')
        self.centerEdit = QLabel(str(self.storage.center))

        grid.addWidget(self.centerTitle, 3, 0)
        grid.addWidget(self.centerEdit, 3, 1)

        self.indexTitle = QLabel('index')
        self.indexEdit = QLineEdit()
        self.indexEdit.setText(str(self.storage.current_index))
        self.indexButton = QPushButton('Go To')
        self.indexButton.clicked.connect(self.main_window.leftChangeIndex)

        
        grid.addWidget(self.indexTitle,4, 0)
        grid.addWidget(self.indexEdit,4, 1)
        grid.addWidget(self.indexButton,4, 2)


        self.setLayout(grid)
        self.show()

class OpenGLWidget(QGLWidget):
    def __init__(self, storage = None, main_window = None):
        super(OpenGLWidget,self).__init__()
        self.storage = storage
        self.main_window = main_window

        self.c = Communicate()
        self.c.rightChangeIndex.connect(self.main_window.rightChangeIndex)
        self.c.updateParams.connect(self.main_window.updateParams)

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

        self.c.updateParams.emit()

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
        # otherwise right widget won't be focused on
        self.setFocusPolicy(Qt.StrongFocus)

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
            self.c.updateParams.emit()
        if e.key() == QtCore.Qt.Key_D:
            self.storage.eye, self.storage.up = transform.left(-5.0, self.storage.eye, self.storage.up, self.storage.center)
            self.updateGL()
            self.c.updateParams.emit()
        if e.key() == QtCore.Qt.Key_W:
            self.storage.eye, self.storage.up = transform.up(5.0, self.storage.eye, self.storage.up, self.storage.center)
            self.updateGL()
            self.c.updateParams.emit()
        if e.key() == QtCore.Qt.Key_S:
            self.storage.eye, self.storage.up = transform.up(-5.0, self.storage.eye, self.storage.up, self.storage.center)
            self.updateGL()
            self.c.updateParams.emit()


        if e.key() == QtCore.Qt.Key_N:
            self.storage.eye, self.storage.up = transform.near(2, self.storage.eye, self.storage.up, self.storage.center)
            self.updateGL()
            self.c.updateParams.emit()
        if e.key() == QtCore.Qt.Key_M:
            self.storage.eye, self.storage.up = transform.near(-2, self.storage.eye, self.storage.up, self.storage.center)
            self.updateGL()
            self.c.updateParams.emit()


        if e.key() == QtCore.Qt.Key_R:
            self.reset()
            self.updateGL()
        if e.key() == QtCore.Qt.Key_T:
            self.storage.is_target = ~ self.storage.is_target
            self.updateGL()


        if e.key() == QtCore.Qt.Key_Left:
            self.storage.current_index -= 1
            self.storage.current_index = self.storage.current_index % len(self.storage.expressions)

            self.c.rightChangeIndex.emit()

        if e.key() == QtCore.Qt.Key_Right:
            self.storage.current_index += 1
            self.storage.current_index = self.storage.current_index % len(self.storage.expressions)
            
            self.c.rightChangeIndex.emit()
        
        if e.key() == QtCore.Qt.Key_Down:
            self.storage.current_index = random.randint(0, len(self.storage.expressions) - 1)
            
            self.c.rightChangeIndex.emit()

 
if __name__ == "__main__":
    app = QApplication(sys.argv)

    #storage
    storage = GlobalStorage()

    win = MainWindow(storage=storage)
    win.show()
    sys.exit(app.exec_())

    print('HELP TEXT:')
    print('\n ----------------------------- \n')
    print ('wasd : rotate the view point')
    print ('nm : drag the view point nearer or further')
    print ('r : reset eye and view direction')
    print ('t : toggle to view predicted model and target model')
    print ('Left and Right Key : view next model\'s voxel representation')
    print('\n ----------------------------- \n')
