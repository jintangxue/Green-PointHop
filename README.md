# A Tiny Machine Learning Model for Point Cloud Object Classification
Created by Min Zhang, Jintang Xue, Pranav Kadam, Hardik Prajapati, Shan Liu, C-C Jay Kuo from University of Southern California.

![introduction](https://github.com/jintangx/Green-PointHop/blob/master/doc/intro.png)

### Introduction
This work is an official implementation of our [arXiv tech report](https://arxiv.org/abs/2303.10898). We proposed a tiny and explainable machine learning method for point cloud, called the Green-PointHop method.

The design of a tiny machine learning model, which can be deployed in mobile and edge devices, for point cloud object classification is investigated in this work. To achieve this objective, we replace the multi-scale representation of a point cloud object with a single-scale representation for complexity reduction, and exploit rich 3D geometric information of a point cloud object for performance improvement. The proposed solution is named Green-PointHop due to its low computational complexity. We evaluate the performance of Green-PointHop on ModelNet40 and ScanObjectNN two datasets. Green-PointHop has a model size of 64K parameters. It demands 2.3M floating-point operations (FLOPs) to classify a ModelNet40 object of 1024 down-sampled points. Its classification performance gaps against the state-of-the-art DGCNN method are 3% and 7% for ModelNet40 and ScanObjectNN, respectively. On the other hand, the model size and inference complexity of DGCNN are 42X and 1203X of those of Green-PointHop, respectively.

In this repository, we release code and data for training a Green-PointHop classification network on point clouds sampled from 3D shapes.

### Citation
If you find our work useful in your research, please consider citing:

    @article{zhang2023tiny,
      title={A Tiny Machine Learning Model for Point Cloud Object Classification},
      author={Zhang, Min and Xue, Jintang and Kadam, Pranav and Prajapati, Hardik and Liu, Shan and Kuo, C-C Jay},
      journal={arXiv preprint arXiv:2303.10898},
      year={2023}
    }
### Installation

The code has been tested with Python 3.7. You may need to install h5py, pytorch, sklearn, pickle and threading packages.

To install h5py for Python:
```bash
sudo apt-get install libhdf5-dev
sudo pip install h5py
```

### Usage
To train a single model without feature selection and ensemble to classify point clouds sampled from 3D shapes:

    python3 train.py

After the above training, we can evaluate the single model.

    python3 evaluate.py

Log files and network parameters will be saved to `log` folder.

Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in `data/modelnet40_ply_hdf5_2048` specifying the ids of shapes in h5 files.
