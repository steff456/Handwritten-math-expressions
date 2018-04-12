import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from skimage import io

def calculate_hog(route_image, visualize = False):
    img = io.imread(route_image, as_grey=True)
    if visualize:
        fd, hog_image = hog(img, visualise=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()
        return hog_image
    else:
        hog_image = hog(img)
        return hog_image
