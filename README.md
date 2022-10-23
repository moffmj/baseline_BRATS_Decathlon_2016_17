# baseline_BRATS_Decathlon_2016_17

Baseline training for a 3D CNN. Using UNet and full 3D convolutions for brain tumour segmentation on the BRATS 2016/17 dataset. Pre-processing normalises the MRI scans to intensity with 0 mean and 1 std. It also merges labels so tumour core and enhancing tumour become one "pathology" label and edema is removed entirely for the purpose of multi-task learning to potentially be applied to different brain conditions e.g. MS, Dementia, TBI.
