import numpy as np
import torch
import os
from scipy.spatial import distance_matrix

import h5py
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.io import FortranFile
import matplotlib.pyplot as plt




def cut(x,y,xl,xu,yl,yu):
    x_index = np.logical_and(x > xl, x < xu)
    y_index = np.logical_and(y > yl, y < yu)
    index = np.logical_and(x_index, y_index)
    return index

def load_mesh_points(filename):
    """
    This function Load mesh from a file.
    """
    f = FortranFile(filename, 'r')
    ni, nj = f.read_ints(dtype=np.int32)
    xy = f.read_reals(dtype=np.float64)
    f.close()
    assert xy.size == ni*nj*2

    xy = xy.reshape(ni, nj, 2, order='F')
    ni = (ni//8)*8
    nj = (nj//8)*8
    xy = xy[:ni, :nj, :]

    mesh = xy.reshape(ni*nj, 2, order='F')

    xy = xy.reshape(ni, nj, 2)
    x = xy[:,0,0]
    y = xy[:,0,1]
    index = cut(x, y, -0.5, 1, -0.5, 0.5)
    boundary = xy[index, 0, :]

    return ni, nj, mesh, boundary

def load_solution_data(filename):
    """
    This function Load solution information from a file.
    """
    f = FortranFile(filename, 'r')
    ni, nj = f.read_ints(dtype=np.int32)
    f.read_reals(dtype=np.float64)
    data = f.read_reals(dtype=np.float64)
    f.close()
    assert data.size == ni*nj*4

    data = data.reshape(4, nj, ni)
    ni = (ni//8)*8
    nj = (nj//8)*8
    data = data[:, :nj, :ni]

    return data.reshape(4, -1).transpose()




path = "CFDdata_252_Paper/s814_Re_1/aoa_00"
path2 = "cfd_data/"
ni, nj, mesh_full, boundary = load_mesh_points(path + "/fort.9")

# np.savetxt(path2+"mesh_full.txt", mesh_full.reshape(-1))

# plt.scatter(mesh[:,0], mesh[:,1],  s=1,  alpha=1)
# plt.title('Matplot scatter plot')
# plt.legend(loc=2)
# plt.show()

mask = cut(mesh_full[:,0], mesh_full[:,1], -0.5,1.5,-0.5,0.5)
index = np.array(range(ni * nj))
index = index[mask]
# np.savetxt(path2+"mask.txt", mask)
# np.savetxt(path2+"index.txt", index)

mesh = mesh_full[index, :]
# np.savetxt(path2+"mesh.txt", mesh.reshape(-1))

## 2 compute the distance matrix

distance_mat_full = distance_matrix(mesh_full, boundary)
distance_full = np.min(distance_mat_full, axis=1)
np.savetxt(path2+"sdf_full.txt", distance_full)

# plt.scatter(mesh[:,0], mesh[:,1], c=distance, s=10, cmap='hsv', alpha=10)
# plt.title('Matplot scatter plot')
# plt.legend(loc=2)
# plt.show()
print(distance_full.shape, mesh_full.shape, index.shape, mask.shape)


for aoa in range(0, 21):

    if aoa < 10:
        folder = "CFDdata_252_Paper/s814_Re_1/aoa_" + "0" + str(aoa)
    else:
        folder = "CFDdata_252_Paper/s814_Re_1/aoa_" + str(aoa)

    filename = folder + "/fort.9"



    # ### save
    new_path = "cfd_data/" + str(aoa)

    if not os.path.exists(new_path):
        os.mkdir(new_path)
        print("Directory ", new_path, " Created ")
    else:
        print("Directory ", new_path, " already exists")


    solution = load_solution_data(folder + "/fort.8")#[index, :]


    np.savetxt(new_path + "/solution_full.txt", solution.reshape(-1))


    print(aoa)
