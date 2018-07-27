from glm import glm
import transform

a = glm.vec3(1,0,0)

b = glm.mat3(a=1,b=2,d=4,g=-2)
print(b)

#c = b + b
#print(c)

#d = 3 * b
#print(d)


R = transform.rotate(90.0, a)
print(R)