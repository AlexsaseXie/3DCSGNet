import math

class vec3:
    def __init__(self,x:float = 0,y:float = 0,z:float = 0):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self):
        return 'x:' + str(self.x) + ', y:' + str(self.y) +', z:' + str(self.z)

    def __getitem__(self,index):
        if (index == 0):
            return self.x
        elif (index == 1):
            return self.y
        elif (index == 2):
            return self.z
        else:
            return None

    def __setitem__(self, index, value):
        if (index == 0):
            self.x = value
        elif (index == 1):
            self.y = value
        elif (index == 2):
            self.z = value

    def normalize(self):
        length = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

        temp = vec3(0,0,0)
        temp.x = self.x / length
        temp.y = self.y / length
        temp.z = self.z / length

        return temp

    def dot(self, b):
        return self.x * b.x + self.y * b.y + self.z * b.z

    def cross(self, b):
        temp = vec3(0,0,0)
        temp[0] = self[1] * b[2] - self[2] * b[1]
        temp[1] = -self[0] * b[2] + self[2] * b[0]
        temp[2] = self[0] * b[1] - self[1] * b[0]
        return temp 

class vec4:
    def __init__(self,x:float = 0,y:float = 0,z:float = 0,w: float = 1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    def __str__(self):
        return 'x:' + str(self.x) + ', y:' + str(self.y) +', z:' + str(self.z) + 'under w=' + str(self.w)

    def __getitem__(self,index):
        if (index == 0):
            return self.x
        elif (index == 1):
            return self.y
        elif (index == 2):
            return self.z
        elif (index == 3):
            return self.w
        else:
            return None

    def __setitem__(self, index, value):
        if (index == 0):
            self.x = value
        elif (index == 1):
            self.y = value
        elif (index == 2):
            self.z = value
        elif (index == 3):
            self.w = value

    def tovec3(self):
        temp = vec3(0,0,0)
        temp.x = self.x / self.w
        temp.y = self.y / self.w
        temp.z = self.z / self.w

        return temp

    def normalize(self):
        self.x /= self.w
        self.y /= self.w
        self.z /= self.w
        self.w = 1
    
class mat3:
    def __init__(self,a:float = 0, b:float = 0, c:float = 0, 
                d:float = 0, e:float = 0, f:float = 0,
                g:float = 0, h:float = 0, i:float = 0):

        self.item = [[0,0,0] for i in range(3)]
        self.item[0][0] = a
        self.item[0][1] = b
        self.item[0][2] = c
        self.item[1][0] = d
        self.item[1][1] = e
        self.item[1][2] = f
        self.item[2][0] = g
        self.item[2][1] = h
        self.item[2][2] = i 
    
    def identity(self=None):
        return mat3(1,0,0,0,1,0,0,0,1)
    
    def zeros(self=None):
        return mat3()

    def getv(self, index1, index2):
        return self.item[index1][index2]

    def setv(self, index1, index2 , value):
        self.item[index1][index2] = value
    
    def matmul(self,b):
        temp = mat3()

        for i in range(3):
            for j in range(3):
                v = 0

                for z in range(3):
                    v += self.getv(i,z) * b.getv(z,j)

                temp.setv(i,j,v)

        return temp

    def vecmul(self,b):
        temp = vec3()

        for i in range(3):
            v = 0
            for j in range(3):
                v += self.getv(i,j) * b[j]

            temp[i] = v
        
        return temp

    def scalarmul(self,b):
        temp = mat3()

        for i in range(3):
            for j in range(3):
                temp.setv(i,j,b * self.getv(i, j))

        return temp

    def matadd(self,b):
        temp = mat3()

        for i in range(3):
            for j in range(3):
                temp.setv(i, j, self.getv(i,j) + b.getv(i, j))

        return temp

    def __add__(self,b):
        return self.matadd(b)

    def __mul__(self,b):
        if isinstance(b, mat3):
            return self.matmul(b)
        elif isinstance(b, vec3):
            return self.vecmul(b)
        else:
            return self.scalarmul(b)

    def __rmul__(self,b):
        return self * b

    def __str__(self):
        temp = ''
        for i in range(3):
            for j in range(3):
                temp += str(self.item[i][j]) + ' '

            temp += '\n'

        return temp

class mat4:
    def __init__(self,a:float = 0, b:float = 0, c:float = 0, d:float = 0,
                e:float = 0, f:float = 0, g:float = 0, h:float = 0,
                i:float = 0, j:float = 0, k:float = 0, l:float = 0,
                m:float = 0, n:float = 0, o:float = 0, p:float = 0):

        self.item = [[0,0,0,0] for i in range(4)]
        self.item[0][0] = a
        self.item[0][1] = b
        self.item[0][2] = c
        self.item[0][3] = d 

        self.item[1][0] = e
        self.item[1][1] = f
        self.item[1][2] = g
        self.item[1][3] = h 
        
        self.item[2][0] = i
        self.item[2][1] = j
        self.item[2][2] = k
        self.item[2][3] = l 

        self.item[3][0] = m
        self.item[3][1] = n
        self.item[3][2] = o
        self.item[3][3] = p 
    
    def identity(self=None):
        return mat4(a=1,f=1,k=1,p=1)
    
    def zeros(self=None):
        return mat4()

    def getv(self, index1, index2):
        return self.item[index1][index2]

    def setv(self, index1, index2 , value):
        self.item[index1][index2] = value
    
    def matmul(self,b):
        temp = mat4()

        for i in range(4):
            for j in range(4):
                v = 0

                for z in range(4):
                    v += self.getv(i,z) * b.getv(z,j)

                temp.setv(i,j,v)

        return temp

    def vecmul(self,b):
        temp = vec4()

        for i in range(4):
            v = 0
            for j in range(4):
                v += self.getv(i,j) * b[j]

            temp[i] = v
        
        return temp

    def matadd(self,b):
        temp = mat4()

        for i in range(4):
            for j in range(4):
                temp.setv(i, j, self.getv(i,j) + b.getv(i, j))

        return temp

    def scalarmul(self,b):
        temp = mat4()

        for i in range(4):
            for j in range(4):
                temp.setv(i,j,b * self.getv(i, j))

        return temp

    def __str__(self):
        temp = ''
        for i in range(4):
            for j in range(4):
                temp += str(self.item[i][j]) + ' '

            temp += '\n'

        return temp