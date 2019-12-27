import h5py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# path2 = "/Users/lizongyi/Downloads/GNN-PDE/fenics/data_sin_high/2"
# path5 = "/Users/lizongyi/Downloads/GNN-PDE/fenics/data_sin_high/5"
path = "/Users/lizongyi/Downloads/cars/out"

train1 = np.loadtxt(path + "/train_loss.txt")
# train2 = np.loadtxt(path + "/multi_res_connected/train_loss_net16.txt")
test1 = np.loadtxt(path + "/test_loss.txt")
# test2 = np.loadtxt(path + "/multi_res_connected/test_loss_net16.txt")


# test = np.loadtxt(path + "/test_loss_net.txt")
# test_u = np.loadtxt(path + "/test_loss_u_net.txt")
# test_mp = np.loadtxt(path + "/test_loss_mp1_net.txt")
# test_su = np.loadtxt(path + "/test_loss_su_net.txt")
#test_fs = np.loadtxt(path + "/test_loss_fs_net.txt")

# plt.plot(train1, label='train add')
# # plt.plot(train2, label='train connected')
# plt.plot(test1, label='test add')
# # plt.plot(test2, label='test connected')
# # plt.plot(test_u, label='fc + u-net')
# # plt.plot(test_mp, label='edge')
# # plt.plot(test_su, label='sep + u-net')
# #plt.plot(test_fs, label='fc + u-net')
# # plt.legend(loc='upper right')
# plt.show()





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


path1 = "/Users/lizongyi/Downloads/cars/out_multi10"
path2 = "car_data/computed_car_flow/sample_"
shape = [128, 256]
for i in range(21,24):
    filename = path2 + str(i+1) + "/fluid_flow_0002.h5"
    boundary_np = load_boundary(filename, shape).reshape(128,256)
    mask = np.where(boundary_np==0)
    sflow_true = load_flow(filename, shape)
    # plot(sflow_true[:, :, 0])
    # plot(sflow_true[:,:,0] - .05 * boundary_np[:, :])

    filename2 = path1 + "/figure"+str(i)+ ".txt"
    pred = np.loadtxt(filename2).reshape(128, 256, 2)

    # image = boundary_np
    # image[mask] = pred[:,0] + 1
    # image = image -1
    # image = image.reshape(128, 256)
    # print(pred[120,:,0])
    # plot(pred[:, :, 0])
    # plot(pred[:,:,0] - .05 * boundary_np[:, :])
    # sflow_plot = np.sqrt(np.square(sflow_true[:, :, 0]) + np.square(sflow_true[:, :, 1])) - .05 * boundary_np[:, :, 0]



# methods = ['ground truth', 'plain', 'connected', 'connected_mp', 'unet', 'multi']
#
# # Fixing random state for reproducibility
# fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
#                         subplot_kw={'xticks': [], 'yticks': []})
#
# for ax, interp_method in zip(axs.flat, methods):
#     ax.imshow(grid, interpolation=interp_method, cmap='viridis')
#     ax.set_title(str(interp_method))
#
# plt.tight_layout()
# plt.show()

def make_image(data, boundary):
    return  np.sqrt(np.square(data[:, :, 0]) + np.square(data[:, :, 1])) - .05 * boundary[:, :]

# Nr = 3
# Nc = 6
# cmap='viridis'
#
# fig, axs = plt.subplots(Nr, Nc, figsize=(18, 6),
#                         subplot_kw={'xticks': [], 'yticks': []})
#
# images = []
# for row in range(Nr):
#
#     i = row + 23
#     filename = "car_data/computed_car_flow/sample_" + str(i+1) + "/fluid_flow_0002.h5"
#     boundary_np = load_boundary(filename, shape).reshape(128,256)
#     sflow_true = load_flow(filename, shape) * 0.8
#     image = make_image(sflow_true, boundary_np)
#     images.append(axs[row, 0].imshow(image, cmap=cmap))
#
#
#     filename = "/Users/lizongyi/Downloads/cars/out" + "/figure"+str(i)+ ".txt"
#     pred = np.loadtxt(filename).reshape(128, 256, 2)
#     image = make_image(pred, boundary_np)
#     images.append(axs[row, 1].imshow(image, cmap=cmap))
#
#     filename = "/Users/lizongyi/Downloads/cars/out_connect_net1" + "/figure" + str(i) + ".txt"
#     pred = np.loadtxt(filename).reshape(128, 256, 2)
#     image = make_image(pred, boundary_np)
#     images.append(axs[row, 2].imshow(image, cmap=cmap))
#
#     filename = "/Users/lizongyi/Downloads/cars/out_connect_MP" + "/figure" + str(i) + ".txt"
#     pred = np.loadtxt(filename).reshape(128, 256, 2)
#     image = make_image(pred, boundary_np)
#     images.append(axs[row, 3].imshow(image, cmap=cmap))
#
#     filename = "/Users/lizongyi/Downloads/cars/out_unet3" + "/figure" + str(i) + ".txt"
#     pred = np.loadtxt(filename).reshape(128, 256, 2)
#     image = make_image(pred, boundary_np)
#     images.append(axs[row, 4].imshow(image, cmap=cmap))
#
#     filename = "/Users/lizongyi/Downloads/cars/out_multi" + "/figure" + str(i) + ".txt"
#     pred = np.loadtxt(filename).reshape(128, 256, 2)
#     image = make_image(pred, boundary_np)
#     images.append(axs[row, 5].imshow(image, cmap=cmap))
#
# # Find the min and max of all colors for use in setting the color scale.
# vmin = min(image.get_array().min() for image in images)
# vmax = max(image.get_array().max() for image in images)
# norm = colors.Normalize(vmin=vmin, vmax=vmax)
# for im in images:
#     im.set_norm(norm)
#
# # fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1)
#
#
# # Make images respond to changes in the norm of other images (e.g. via the
# # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
# # recurse infinitely!
# def update(changed_image):
#     for im in images:
#         if (changed_image.get_cmap() != im.get_cmap()
#                 or changed_image.get_clim() != im.get_clim()):
#             im.set_cmap(changed_image.get_cmap())
#             im.set_clim(changed_image.get_clim())
#
#
# for im in images:
#     im.callbacksSM.connect('changed', update)
#
# plt.tight_layout()
# plt.show()



def make_image2(data):
    data = data.reshape(-1,3)
    return data[:,0]
    # return  np.sqrt(np.square(data[:, 0]) + np.square(data[:, 1]) + np.square(data[:, 2]))

# if aoa < 10:
#     folder = "CFDdata_252_Paper/s814_Re_1/aoa_" + "0" + str(aoa)
# else:
#     folder = "CFDdata_252_Paper/s814_Re_1/aoa_" + str(aoa)

mesh = np.loadtxt("cfd_data/mesh.txt").reshape(-1, 2)

# i = 0
# filename = "/Users/lizongyi/Downloads/cfd/out_mp/figure"  + str(i*2+14) +".txt"
# solution = np.loadtxt(filename)
# image = make_image2(solution)
# plt.scatter(mesh[:,0], mesh[:,1], c=image, s=10, cmap='viridis', alpha=10)
# plt.show()

Nr = 3
Nc = 5
cmap='viridis'

fig, axs = plt.subplots(Nr, Nc, figsize=(18, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

images = []
for row in range(Nr):

    i = row*3 + 14
    filename = "cfd_data/" + str(i) + "/solution.txt"
    image = np.loadtxt(filename).reshape(-1,4)[:,-2]
    images.append(axs[row, 0].scatter(mesh[:,0], mesh[:,1], c=image, s=10, cmap='viridis', alpha=10))


    filename = "/Users/lizongyi/Downloads/cfd/out/figure"  + str(i) +".txt"
    image = np.loadtxt(filename).reshape(-1, 3)[:, -2]
    images.append(axs[row, 1].scatter(mesh[:,0], mesh[:,1], c=image, s=10, cmap='viridis', alpha=10))

    filename = "/Users/lizongyi/Downloads/cfd/out_mp/figure" + str(i) + ".txt"
    image = np.loadtxt(filename).reshape(-1, 3)[:, -2]
    images.append(axs[row, 2].scatter(mesh[:,0], mesh[:,1], c=image, s=10, cmap='viridis', alpha=10))

    filename = "/Users/lizongyi/Downloads/cfd/out_multi/figure" + str(i) + ".txt"
    image = np.loadtxt(filename).reshape(-1, 3)[:, -2]
    images.append(axs[row, 3].scatter(mesh[:,0], mesh[:,1], c=image, s=10, cmap='viridis', alpha=10))

    filename = "/Users/lizongyi/Downloads/cfd/out_multi_mp/figure" + str(i) + ".txt"
    image = np.loadtxt(filename).reshape(-1, 3)[:, -2]
    images.append(axs[row, 4].scatter(mesh[:,0], mesh[:,1], c=image, s=10, cmap='viridis', alpha=10))

# Find the min and max of all colors for use in setting the color scale.
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)

# fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1)


# Make images respond to changes in the norm of other images (e.g. via the
# "edit axis, curves and images parameters" GUI on Qt), but be careful not to
# recurse infinitely!
def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())


for im in images:
    im.callbacksSM.connect('changed', update)

plt.tight_layout()
plt.show()
