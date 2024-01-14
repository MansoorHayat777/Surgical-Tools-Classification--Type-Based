
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt


#clipLimit – This parameter sets the threshold for contrast limiting. The default value is 40.
# tileGridSize – This sets the number of tiles in the row and column. By default this is 8×8. It is used while the image is divided into tiles for applying CLAHE.
def hisEqulColor(img):
  ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
  channels = cv2.split(ycrcb)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  channels[0] = clahe.apply(channels[0])
  cv2.merge(channels, ycrcb)
  cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
  return img


x = cv2.imread("/home/sayak/Documents/personal/vit/study/thirdY6thSem/tarp/surgical_tool_seg/Mask_RCNN/seggy/test/img_118_raw.png")
original_image = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
# plt.hist(original_image.flat, bins=100, range=(0, 255))
img2 = hisEqulColor(original_image)

# plt.hist(img2.flat, bins=100, range=(0, 255))

#Generating the histogram of the original image
hist_c,bins_c = np.histogram(original_image.flatten(),256,[0,256])

#Generating the cumulative distribution function of the original image
cdf_c = hist_c.cumsum()
cdf_c_normalized = cdf_c * hist_c.max()/ cdf_c.max()

#Generating the histogram of the image after applying CLAHE
hist_c_clahe, bins_c_clahe = np.histogram(img2.flatten(),256,[0,256])

#Generating the cumulative distribution function of the original image
cdf_c_clahe = hist_c_clahe.cumsum()
cdf_c_clahe_normalized = cdf_c_clahe * hist_c_clahe.max()/ cdf_c_clahe.max()

#Plotting the Original and Histogram Equalized Image, Histogram and CDF
fig, axs = plt.subplots(2, 2)

axs[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axs[0, 0].axis('off')
axs[0, 1].set_title('Image after CLAHE')



axs[1, 1].plot(cdf_c_clahe_normalized, color = 'b')
axs[1, 1].hist(img2.flatten(),256,[0,256], color = 'r')
axs[1, 1].legend(('cdf_clahe','histogram_clahe'), loc = 'upper left')


# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# cv2.imshow('enhance',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


