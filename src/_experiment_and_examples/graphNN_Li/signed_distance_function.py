import numpy as np
import torch
from scipy.spatial import distance_matrix

import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_flow(filename, shape):
  stream_flow = h5py.File(filename, 'r')
  flow_state_vel = np.array(stream_flow['Velocity_0'][:])
  flow_state_vel = flow_state_vel.reshape([shape[0], shape[1]+128, 3])[0:shape[0],0:shape[1],0:2]
  stream_flow.close()
  return flow_state_vel

def load_boundary(filename, shape):
  stream_boundary = h5py.File(filename, 'r')
  boundary_cond = np.array(stream_boundary['Gamma'][:])
  boundary_cond = boundary_cond.reshape([shape[0], shape[1]+128, 1])[0:shape[0],0:shape[1],:]
  stream_boundary.close()
  return boundary_cond

def plot(image):
    plt.imshow(image)
    plt.colorbar()
    plt.show()

## 0 # read raw boundary
n_x = 128
n_y = 256
path = "car_data/computed_car_flow/sample_"
shape = [n_x, n_y]


for car in range(1, 29):
    filename =  path  + str(car) + "/fluid_flow_0002.h5"
    boundary = load_boundary(filename, shape).reshape(n_x, n_y)
    solution = load_flow(filename, shape).reshape(n_x, n_y, 2)
    # plot(boundary)

    ## 1 find the edge
    edge = np.zeros((n_x, n_y), dtype=np.uint8)
    for i in range(n_x):
        for j in range(n_y):
            if boundary[i,j] == 1:
                is_edge = False
                for a in range(-1,2):
                    for b in range(-1,2):
                        x = i + a
                        y = j + b
                        if x >= 0 and x <=n_x-1 and y>=0 and y <= n_y-1:
                            if boundary[x,y] == 0:
                                is_edge = True
                if is_edge:
                    edge[i,j] = 1

    # plot(edge)
    edge_index = np.where(edge.reshape(-1) == 1)

    ## 2 compute the distance matrix
    xs = np.array(range(n_y))
    ys = np.array(range(n_x))
    mesh_position = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_position = mesh_position[edge_index]

    distance_mat = distance_matrix(mesh_position, edge_position)
    distance = np.min(distance_mat, axis=1).reshape(n_x,n_y)
    sign = boundary * -2 + 1
    signed_distance = sign * distance
    # plot(signed_distance)


    ### save
    filename =  path  + str(car) + "/sdf.txt"
    np.savetxt(path  + str(car) + "/sdf.txt", signed_distance.reshape(-1))
    np.savetxt(path  + str(car) + "/boundary.txt", boundary.reshape(-1))
    np.savetxt(path + str(car) + "/solution.txt", solution.reshape(-1))

    print(car)
