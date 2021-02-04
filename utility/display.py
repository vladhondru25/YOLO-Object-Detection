import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_boxes(image, bboxes):
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(image.permute(1, 2, 0))

    # Create a Rectangle patch
    for bbox in bboxes:
        rect = patches.Rectangle((bbox[2],bbox[3]),bbox[4],bbox[5],linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()