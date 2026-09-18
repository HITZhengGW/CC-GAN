import numpy as np
import os
import nibabel as nib
from scipy.ndimage import zoom

def CCTA_split():
    file_name = "./0.nii.gz"
    img_nifti = nib.load(file_name)
    voxels_space = img_nifti.header['pixdim'][1:4]
    img = img_nifti.get_fdata()
    data = np.array(img)

    data = zoom(data, (voxels_space[0], voxels_space[1], voxels_space[2]), order=0, mode='nearest') > 0
    pos = np.where(data>0.5)
    xyzs = [pos[0], pos[1], pos[2]]

    v_min = np.min(xyzs[0])
    v_max = np.max(xyzs[0])
    xyzs[0] = xyzs[0] - v_min
    x_diff = v_max - v_min
    # print(x_diff)

    v_min = np.min(xyzs[1])
    v_max = np.max(xyzs[1])
    xyzs[1] = xyzs[1] - v_min
    y_diff = v_max - v_min
    # print(y_diff)

    v_min = np.min(xyzs[2])
    v_max = np.max(xyzs[2])
    xyzs[2] = xyzs[2] - v_min
    z_diff = v_max - v_min
    # print(z_diff)

    if x_diff < 128 and y_diff < 128 and z_diff < 128:
        x_gap = 128 - (x_diff + 1)
        y_gap = 128 - (y_diff + 1)
        z_gap = 128 - (z_diff + 1)

        xyzs[0] = xyzs[0] + int(x_gap/2)
        xyzs[1] = xyzs[1] + int(y_gap/2)
        xyzs[2] = xyzs[2] + int(z_gap/2)

        data = np.zeros((128,128,128))
        data[xyzs[0],xyzs[1],xyzs[2]] = 1

        w, h, d = data.shape
        coords = []
        flag = False
        for i in range(w):
            if flag:
                break
            for j in range(h):
                if flag:
                    break
                for k in range(d):
                    if data[i,j,k] > 0:
                        coords.append([i,j,k])
                        flag = True
                        break

        for [x,y,z] in coords:
            for cx in [x-1,x,x+1]:
                for cy in [y-1,y,y+1]:
                    for cz in [z-1,z,z+1]:
                        c_coord = [cx,cy,cz]
                        if not (c_coord in coords):
                            if cx > -1 and cx < w:
                                if cy > -1 and cy < h:
                                    if cz > -1 and cz < d:
                                        if data[cx,cy,cz] > 0:
                                            coords.append(c_coord)

        coords = np.transpose(np.array(coords))
        data[coords[0],coords[1],coords[2]] = 0
        os.makedirs("./split_one", exist_ok=True)
        np.save("./split_one/data", data.astype('int8'))

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # xyzs = np.where(data>0.5)
        # ax.scatter(xyzs[0], xyzs[1], xyzs[2], marker='.')
        # plt.show()

        data = data*0
        data[coords[0],coords[1],coords[2]] = 1
        os.makedirs("./split_two", exist_ok=True)
        np.save("./split_two/data", data.astype('int8'))   

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # xyzs = np.where(data>0.5)
        # ax.scatter(xyzs[0], xyzs[1], xyzs[2], marker='.')
        # plt.show()

    else:
        print('Failed to split for this data!')


if __name__ == "__main__":
    CCTA_split()
