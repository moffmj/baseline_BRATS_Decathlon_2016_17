import image_normalisation
import numpy as np

if __name__ == "__main__":

    data_path = "/Users/mattmoffat/4yp_datasets/Task01_BrainTumour/labelsTr"
    save_path = "/Users/mattmoffat/4yp_datasets/BRATS_merged_labels"

    # labels
	# "0": "background", 
	# "1": "edema",
	# "2": "non-enhancing tumor",
	# "3": "enhancing tumour"
    # merge 2 & 3 as just "pathology" for multi-task purposes and remove edema as a segmentation label

    number_of_files = 484

    for i in range(1,number_of_files+1):
        a = str(i).zfill(3)
        print (a + " of " + str(number_of_files))
        
        load_name = "BRATS_" + a + ".nii.gz"
        save_name = "BRATS_" + a + "merged.nii.gz"

        nifti_file = image_normalisation.load_nifti(data_path=data_path, file_name=load_name)
        nifti_array = image_normalisation.nifti_to_array(nifti_file)

        nifti_array[nifti_array==1] = 0
        nifti_array[nifti_array==2] = 1
        nifti_array[nifti_array==3] = 1

        image_normalisation.save_arr(nifti_array,save_path,save_name)
    
    