"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import os
from data.base_dataset import BaseDataset
import nibabel as nib
import numpy as np


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    # if opt.loader == 'default':
    data_loader = CustomDatasetDataLoader(opt)
    # elif opt.loader == "gazi":
        # data_loader = GaziDatasetDataLoader(opt)
    #elif opt.loader = "brats":
    # data_loader = ...
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

# class GaziDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         print(root_dir)
#         self.data_folders = self._get_valid_data_folders()

#     def __len__(self):
#         return len(self.data_folders)

#     def __getitem__(self, idx):
#         folder_name = self.data_folders[idx]
#         # Load your data from the folder and return it as a dictionary or tuple
#         flair_imgs, t1w_imgs, t2w_imgs, gadolinium_t1w_imgs = self._load_data_from_folder(folder_name)

#         batch_x, batch_y = [], []

#         assert(t1w_imgs.shape[2] == t2w_imgs.shape[2] == flair_imgs.shape[2] == gadolinium_t1w_imgs.shape[2])
#         no_of_sequences = t1w_imgs.shape[2]

#         for i in range(0, no_of_sequences):
#             t1w_img = t1w_imgs[..., i]
#             t2w_img = t2w_imgs[..., i]
#             flair_img = flair_imgs[..., i]
#             gadolinium_t1w_img = gadolinium_t1w_imgs[..., i]

#             assert(t1w_img.shape == t2w_img.shape == flair_img.shape == gadolinium_t1w_img.shape)

#             concatenated_img = np.concatenate([t1w_img[..., np.newaxis], t2w_img[..., np.newaxis], flair_img[..., np.newaxis]], axis=-1)
#             batch_x.append(concatenated_img)
#             batch_y.append(gadolinium_t1w_img)

#         return batch_x, batch_y

#     def _get_valid_data_folders(self):
#         data_folders = []
#         for folder_name in os.listdir(self.root_dir):
#             folder_path = os.path.join(self.root_dir, folder_name)
#             anat_path = os.path.join(folder_path, "anat")
#             if os.path.isdir(folder_path) and os.path.isdir(anat_path) and self._has_optional_files(folder_path, anat_path):
#                 data_folders.append(folder_path)
#         return data_folders

#     def _has_optional_files(self, folder_path, anat_path):
#         base_name = os.path.basename(folder_path)
#         optional_files = [f'{base_name}_ce-GADOLINIUM_T1w.json', f'{base_name}_ce-GADOLINIUM_T1w.nii.gz']
#         for file_name in optional_files:
#             file_path = os.path.join(anat_path, file_name)
#             if not os.path.exists(file_path):
#                 return False
#         return True

#     def _load_data_from_folder(self, folder_name):
#         # Implement your data loading logic here
#         folder_path = os.path.join(self.root_dir, folder_name)

#         base_name = os.path.basename(folder_path)

#         folder_path = os.path.join(folder_path, "anat")
        
#         # Load mandatory files
#         t1w_path = os.path.join(folder_path, f"{base_name}_T1w.nii.gz")
#         flair_path = os.path.join(folder_path, f"{base_name}_FLAIR.nii.gz")
#         t2w_path = os.path.join(folder_path, f"{base_name}_T2w.nii.gz")
#         t1w_img = nib.load(t1w_path).get_fdata()
#         flair_img = nib.load(flair_path).get_fdata()
#         t2w_img = nib.load(t2w_path).get_fdata()

#         # Load optional files if available
#         gadolinium_t1w_path = os.path.join(folder_path, f"{base_name}_ce-GADOLINIUM_T1w.nii.gz")
#         gadolinium_t1w_img = nib.load(gadolinium_t1w_path).get_fdata()

#         return flair_img, t1w_img, t2w_img, gadolinium_t1w_img

# class GaziDatasetDataLoader():
#     """Wrapper class of Dataset class that performs multi-threaded data loading"""

#     def __init__(self, opt):
#         """Initialize this class

#         Step 1: create a dataset instance given the name [dataset_mode]
#         Step 2: create a multi-threaded data loader.
#         """
#         self.opt = opt
#         self.dataset = GaziDataset(opt.dataroot)
#         print(len(self.dataset))
#         print("Dataset [%s] was created" % type(self.dataset).__name__)
#         self.dataloader = DataLoader(
#             self.dataset,
#             batch_size=opt.batch_size,
#             shuffle=not opt.serial_batches,
#             num_workers=int(opt.num_threads))

#     def load_data(self):
#         return self

#     def __len__(self):
#         """Return the number of data in the dataset"""
#         return len(self.dataset)

#     def __iter__(self):
#         """Return a batch of data"""
#         for i, data in enumerate(self.dataloader):
#             if i * self.opt.batch_size >= self.opt.max_dataset_size:
#                 break
#             yield data