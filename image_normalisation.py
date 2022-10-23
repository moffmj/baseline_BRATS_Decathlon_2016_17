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

def normalise(arr):
    normed = (arr - np.mean(arr, axis=(0,1,2)))/np.std(arr,axis=(0,1,2))
    return normed

def save_arr(arr, save_path, file_name):
    new_image = nib.Nifti1Image(arr, affine=np.eye(4))
    save_path = os.path.join(save_path,file_name)
    nib.save(new_image, save_path)

if __name__ == "__main__":

    number_of_files = 484
    data_path = "/Users/mattmoffat/4yp_datasets/Task01_BrainTumour/imagesTr"
    save_path = "/Users/mattmoffat/4yp_datasets/BRATS_Normalised"

    for i in range(1,number_of_files+1):
        a = str(i).zfill(3)
        print (a + " of " + str(number_of_files))
        
        load_name = "BRATS_" + a + ".nii.gz"
        save_name = "BRATS_" + a + "_norm.nii.gz"

        nifti_file = load_nifti(data_path=data_path, file_name=load_name)
        nifti_array = nifti_to_array(nifti_file)
        normed = normalise(nifti_array)
        save_arr(normed, save_path, save_name)

    
    

    
