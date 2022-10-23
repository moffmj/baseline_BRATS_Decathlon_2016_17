import nibabel as nib
import os
from nibabel.testing import data_path
import numpy as np

def load_nifti(data_path, file_name):
    file_path = os.path.join(data_path, file_name)
    img = nib.load(file_path)
    return img

def nifti_to_array(img):
    nifti_array = np.array(img.dataobj)
    return nifti_array



if __name__ == "__main__":

    data_path_standard = "/Users/mattmoffat/4yp_datasets/Task01_BrainTumour/imagesTr"
    load_name_standard = "BRATS_001.nii.gz"

    data_path_normed = "/Users/mattmoffat/4yp_datasets/BRATS_Normalised"
    load_name_normed = "BRATS_001_norm.nii.gz"

    nifti_file_standard = load_nifti(data_path=data_path_standard, file_name=load_name_standard)
    nifti_array_standard = nifti_to_array(nifti_file_standard)

    nifti_file_normed = load_nifti(data_path=data_path_normed, file_name=load_name_normed)
    nifti_array_normed = nifti_to_array(nifti_file_normed)

    print(nifti_array_standard.shape)
    print(nifti_array_normed.shape)

    mean_standard = np.mean(nifti_array_standard,axis=(0,1,2))
    std_standard = np.std(nifti_array_standard,axis=(0,1,2))
    print(mean_standard)
    print(std_standard)

    mean_normed = np.mean(nifti_array_normed,axis=(0,1,2))
    std_normed = np.std(nifti_array_normed,axis=(0,1,2))
    print(mean_normed)
    print(std_normed)


    
    

    
