import os
from pydicom import dcmread
import numpy as np
import torch
from glob import glob
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm


def visimgs(savepath="", show=False, **images):
    """Plot images."""
    n = len(images)
    
    # Close all previous plt sessions and clear canvases.
    plt.close()
    plt.clf()
    plt.figure(figsize=(16, 5))
    
    # Plot images with their given parameter names.
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image, cmap="Greys_r")

    # Save image to file.
    plt.savefig(savepath, bbox_inches='tight')
    
    # Show image if flag is set.
    if show:
        plt.show()

# Setup CUDA device.
device = torch.device("cuda")

# Load and setup model.
model = torch.load("./best.pth")
model.to(device)
model.eval()

# Look into dicom data.
dcms = glob("./us_data/**/**/*.dcm")

# Iterate through all dicom images.
for dcm_path in tqdm(dcms):
    dcm_image = dcmread(dcm_path)  # Read dicom object.

    # If there is no image data in the dicom file, skip it.
    if 'Image Storage' not in dcm_image.SOPClassUID.name:
        continue  # to skip the rest of the loop
    else:
        image = dcm_image.pixel_array  # Read image data.
        
        try:  # This try-catch rule is for cases where the channel configuration is not straghtforward.
            
            # Isolate some naming patameters.
            parentdir = dcm_path.split("/")[2]
            childdir = dcm_path.split("/")[3]
            imgname = dcm_path.split("/")[4]
            
            # Create save dir.
            os.makedirs(f"./us_data_pred/{parentdir}/{childdir}", exist_ok=True)
            
            # Convert image to greyscale and resize to model input dimensions.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (800, 544))
            
            # Write original image to save dir.
            cv2.imwrite(f"./us_data_pred/{parentdir}/{childdir}/{imgname.replace('dcm', 'png')}", image)
            
            # Format image to be a torch tensor and get model prediction.
            inpt_image = torch.Tensor(image[ np.newaxis, np.newaxis, :, :] / 255.)
            pred = model(inpt_image.to(device))
            
            # Visualize and save model predictions to save dir.
            visimgs(savepath=f"./us_data_pred/{parentdir}/{childdir}/{imgname.replace('.dcm', 'result.png')}",
                    show=False,
                    original=inpt_image.clone().detach().cpu().numpy().squeeze(),
                    prediction=pred.clone().detach().cpu().numpy().squeeze())
            
        except Exception:  # Exception handling.
            print(f"Could not process {dcm_path}")
            with open("didnotread.txt", 'a') as f:
                f.write(f"{dcm_path}\n")

