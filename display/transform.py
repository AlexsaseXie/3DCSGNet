from glm import glm
import math


def rotate(degrees: float, axis):
    theta = degrees * math.pi / 180.0

    R1 = glm.mat3.identity()
    
    R2 = glm.mat3 (axis.x * axis.x, axis.x * axis.y, axis.x * axis.z, 
    axis.x * axis.y, axis.y * axis.y, axis.y * axis.z,
    axis.x * axis.z, axis.y * axis.z, axis.z * axis.z)

    R3 = glm.mat3(0, -axis.z, axis.y, 
                axis.z, 0, -axis.x,
                -axis.y, axis.x, 0)

    R = R1 * (math.cos(theta)) + R2 * (1-math.cos(theta)) + R3 * (math.sin(theta))

    return R

def left(degrees: float, eye, up):
    eye = rotate(degrees, up) * eye

    return eye, up

def up(degrees: float, eye, up):
    x = up.cross(eye)
    x = x.normalize()
    
    eye = rotate(degrees, x) * eye

    up = eye.cross(x)
    up = up.normalize()

    return eye, up

def near(dis: float, eye, up):
    length = math.sqrt(eye.x * eye.x + eye.y * eye.y + eye.z * eye.z)

    eye.x -= dis * eye.x / length
    eye.y -= dis * eye.y / length
    eye.z -= dis * eye.z / length

    return eye, up