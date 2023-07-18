import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import h5py
import glob
SAMPLING_BIN = os.path.join(BASE_DIR, 'third_party/mesh_sampling/build/pcsample')

SAMPLING_POINT_NUM = 2048
SAMPLING_LEAF_SIZE = 0.005

shapenet_PATH = '/home/hannah/Thesis/dgcnn/pytorch/shapeNetDataPrep/03001627'
def export_ply(pc, filename):
	vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	for i in range(pc.shape[0]):
		vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
	ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
	ply_out.write(filename)

# Sample points on the obj shape
def get_sampling_command(obj_filename, ply_filename):
    cmd = SAMPLING_BIN + ' ' + obj_filename
    cmd += ' ' + ply_filename
    cmd += ' -n_samples %d ' % SAMPLING_POINT_NUM
    cmd += ' -leaf_size %f ' % SAMPLING_LEAF_SIZE
    return cmd

# --------------------------------------------------------------
# Following are the helper functions to load MODELNET40 shapes
# --------------------------------------------------------------

# Read in the list of categories in MODELNET40
def get_category_names():
    # shape_names_file = os.path.join(MODELNET40_PATH, 'shape_names.txt')
    # shape_names = [line.rstrip() for line in open(shape_names_file)]
    shape_names = ["Chair"]
    return shape_names

# Return all the filepaths for the shapes in MODELNET40 
def get_obj_filenames():
    obj_filelist_file = os.path.join('/home/hannah/Thesis/dgcnn/pytorch/shapeNetDataPrep', 'arm_labels.txt')
    # obj_filenames = [os.path.join(shapenet_PATH, line.rstrip()) for line in open(obj_filelist_file)]
    obj_filenames = [os.path.join(shapenet_PATH, line.rstrip()[:-2]) for line in open(obj_filelist_file)]

    print('Got %d obj files in shapeNet.' % len(obj_filenames))
    return obj_filenames

# Helper function to create the father folder and all subdir folders if not exist
def batch_mkdir(output_folder, subdir_list):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for subdir in subdir_list:
        if not os.path.exists(os.path.join(output_folder, subdir)):
            os.mkdir(os.path.join(output_folder, subdir))

# ----------------------------------------------------------------
# Following are the helper functions to load save/load HDF5 files
# ----------------------------------------------------------------

# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal, 
		data_dtype='float32', label_dtype='uint8', normal_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

# Read numpy array data and label from h5_filename
def load_h5_data_label_normal(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return (data, label, normal)

# Read numpy array data and label from h5_filename
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# ----------------------------------------------------------------
# Following are the helper functions to load save/load PLY files
# ----------------------------------------------------------------

# Load PLY file
def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Load PLY file
def load_ply_normal(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['normal'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Make up rows for Nxk array
# Input Pad is 'edge' or 'constant'
def pad_arr_rows(arr, row, pad='edge'):
    assert(len(arr.shape) == 2)
    assert(arr.shape[0] <= row)
    assert(pad == 'edge' or pad == 'constant')
    if arr.shape[0] == row:
        return arr
    if pad == 'edge':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'edge')
    if pad == 'constant':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'constant', (0, 0))


def plot_pcd(sizes=None, cmap='Greys', zdir='y',
                         xlim=(-0.4, 0.4), ylim=(-0.4, 0.4), zlim=(-0.4, 0.4)):
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d
    pcds = np.load('/home/hannah/Thesis/dgcnn/pytorch/shapeNetDataPrep/03001627/train/1ab42ccff0f8235d979516e720d607b8.npy')
    # pcds = np.load('/home/hannah/Thesis/SP-GAN/models/pcds/chair_1_000022.npy')[0]
    print(pcds.shape)
    pcds = downsampling(pcds)
    print(pcds.shape)
    if sizes is None:
        sizes = [0.2 for i in range(len(pcds))]

    print(len(pcds),pcds.shape)
    x = pcds[:,0]
    y = pcds[:,1]
    z = pcds[:,2]

    fig = plt.figure(figsize=(10,10)) # W,H
    ax = plt.axes(projection='3d')
    # ax.grid()
    elev = 30
    azim = -45
    ax.view_init(elev, azim)
    
    ax.scatter(x, y, z, c = 'r', s = 50, zdir=zdir)
    ax.set_title('3D Scatter Plot')
    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)
    plt.show()
def downsampling(file):
    import numpy as np
    import open3d as o3d
    print("Downsample the point cloud uniformly,choose every 5 points")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(file)
    # downpcd = pcd.voxel_down_sample(voxel_size=0.02) # the number of points are not equal
    downpcd = pcd.uniform_down_sample(5) # this one is good, (3000,3) 
    #The sample is performed in the order of the points with the 0-th point always chosen, not at random

    # o3d.visualization.draw_geometries([downpcd])
    return np.asarray(downpcd.points)

def dataPairLabel():
    import shutil
    labels = []
    labelPath = '/home/hannah/Thesis/dgcnn/pytorch/shapeNetDataPrep/arm_labels.txt'
    destination_dir = '/home/hannah/Thesis/dgcnn/pytorch/shapeNetDataPrep/chair/data'
    partition = ['train','test','val']
    data = []
    datafilename = []
    for line in open(labelPath):
        for i in partition:
            filename = '%s.npy'% line.rstrip()[:-2]
            file = os.path.join('/home/hannah/Thesis/dgcnn/pytorch/shapeNetDataPrep/03001627/%s'%i, filename) #filepath
            if os.path.exists(file):
                npfile = np.load(file) #load numpy array
                npfile = downsampling(npfile) #downsampling
                print("after downsampling, the shape is:",npfile.shape)
                shutil.copy(file,destination_dir) #copy file to another data folder
                labels.append(line.rstrip()[-1]) # save label
                data.append(npfile) #save data
                datafilename.append(filename) #save filename for double check.
    # np.save('/home/hannah/Thesis/dgcnn/pytorch/shapeNetDataPrep/chair/labels/label',labels)
    np.save('/home/hannah/Thesis/dgcnn/pytorch/shapeNetDataPrep/chair/alldata',data)

    files = os.listdir(destination_dir)
    print("the number of labels",len(labels))
    print("the first 10 lables:",labels[:10])
    print("the number of selected files:",len(files))
    print("the first 10 filenames:",datafilename[:10])
    print("all data array shape is :",len(data))

if __name__ == '__main__':

    # obj_filenames = get_obj_filenames()
    # print(obj_filenames[0])
    # ply_filename = "ply1test"
    # cmd = get_sampling_command(obj_filenames[0],ply_filename)
    # print(cmd)
    # plot_pcd()
    # dataPairLabel()
    print("data preprocessing is done")
    data = np.load('/home/hannah/Thesis/dgcnn/pytorch/shapeNetDataPrep/chair/alldata.npy')
    labels = np.load('/home/hannah/Thesis/dgcnn/pytorch/shapeNetDataPrep/chair/label.npy')
    print("data length:",len(data))
    print("the shape of each pcd:",data[0].shape)
    print("number of labels:",len(labels))
   
