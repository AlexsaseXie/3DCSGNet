import deepdish as dd
from src.Utils.train_utils import voxels_from_expressions

# pre-rendered shape primitives in the form of voxels for better performance
primitives = dd.io.load("data/primitives.h5")
expressions = ["cy(48,48,32,8,12)cu(24,24,40,28)+", "sp(48,32,32,8,12)cu(24,24,40,28)+"]

voxels = voxels_from_expressions(expressions, primitives, max_len=7)
print(voxels.shape)

(2, 64, 64, 64)