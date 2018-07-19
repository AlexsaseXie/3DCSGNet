import glm as glm 

# a = glm.vec3(1,1,1)
# b = glm.vec3(0,2,-1)

# c = a.dot(b)
# print(c)

# d = a.cross(b)
# print(d)

# e = a.normalize()
# print(e)

# e[0] = 3

# print(e)

# a = glm.vec4(1,1,1,2)
# b = a.tovec3()
# print (b)


# a = glm.mat3(1,1,2,3,4,5,0,-1,2)
# b = glm.vec3(1,-2,3)

# c = a.getv(1,2)
# print ('c=', c)

# d = a.vecmul(b)
# print (d)

# print('a:\r\n',a)
# f = a.matmul(a)
# print('f:\r\n',f)

a = glm.mat4.identity(glm.mat4)
a.setv(0,1,1)

for i in range(10):
    a = a.matmul(a)

    print(a)

f = glm.mat3.identity()
print(f)
