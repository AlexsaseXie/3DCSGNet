from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def fake_data():
    a = np.zeros([64,64,64],dtype=bool)

    l = []
    for i in range(100):
        temp = np.random.randint(0,64,size=3)
        l.append(temp)
        a[temp[0],temp[1],temp[2]] = True
    return a, l

def find_points(a):
    l = []
    for i in range(64):
        for j in range(64):
            for k in range(64):
                if a[i,j,k]== True:
                    l.append([i,j,k])
    return l

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


img,_ = fake_data()
print(img)
l = find_points(img)

xs = [lt[0] for lt in l]
ys = [lt[1] for lt in l]
zs = [lt[2] for lt in l] 
ax.scatter(xs, ys, zs, c='r', marker='.')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()