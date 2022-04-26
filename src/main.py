from functions import *

if __name__ == '__main__':

  # Read reference image
  reference_img = cv2.imread("../imgs/plantilla.jpg", cv2.IMREAD_COLOR)

  # Read image to be aligned
  align_img = cv2.imread("../imgs/foto.jpeg", cv2.IMREAD_COLOR)

  #Calculate the homography and store it in h with the corrected img in corrected_img
  corrected_img, h = alignImages(align_img, reference_img)

  # Write aligned image to disk.
  cv2.imwrite("../imgs/aligned.jpg", corrected_img)