import nibabel as nib
import numpy as np
import os

def save_annotation_per_label(path_nifti, path_output, lesion_labels):
    """
    This function processes a NIfTI annotation image containing multiple labeled regions 
    (e.g., different types of lesions) and saves each labeled region as a separate NIfTI file.

    Parameters:
    -----------
    path_nifti : str
        The file path to the input NIfTI annotation image. This image should contain labeled 
        regions where each region is identified by a unique integer value (label).
    
    path_output : str
        The directory path where the function will save the resulting NIfTI files. Each file will 
        correspond to one of the labels in the `lesion_labels` list.
    
    lesion_labels : list of int
        A list of integer labels that you want to isolate from the annotation image. Each label 
        corresponds to a specific type of lesion or annotated region in the image. The function 
        will create and save a separate NIfTI file for each label.

    Returns:
    --------
    None
        The function does not return any values. It directly saves the resulting NIfTI files 
        in the specified output directory.
    """

    # Load the NIfTI annotation image from the specified path
    annotation_img = nib.load(path_nifti)
    
    # Extract the image data into a NumPy array for easy manipulation
    annotation_data = annotation_img.get_fdata()

    # Iterate over each label provided in the lesion_labels list
    for label in lesion_labels:
        # Create a binary mask where the current label is 1, and all other values are 0
        lesion_mask = (annotation_data == label).astype(np.float32)
        
        # Create a new NIfTI image using the binary mask, maintaining the same affine transformation and header as the original image
        lesion_img = nib.Nifti1Image(lesion_mask, affine=annotation_img.affine, header=annotation_img.header)
        
        # Save the new NIfTI image in the output directory, with a filename that includes the label to indicate which lesion type it represents
        nib.save(lesion_img, os.path.join(path_output, f'lesion_type_{label}.nii'))

        print(f"Saved lesion type {label} as a separate NIfTI file.")
