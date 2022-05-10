from functions import *
from filtre import *

if __name__ == '__main__':
    
  # Read face image to be aligned
  face_img = cv2.imread("foto_persona", cv2.IMREAD_COLOR)
  
  # Read reference image
  reference_img = cv2.imread("../imgs/plantilla.jpg", cv2.IMREAD_COLOR)

  # Read filter image to be aligned
  filter_img = cv2.imread("foto_filtro", cv2.IMREAD_COLOR)

  #Calculate the homography and store it in h with the corrected img in corrected_img
  filter_img, h = alignImages(filter_img, reference_img)

  # Write aligned image to disk.
  #cv2.imwrite("../imgs/aligned.jpg", corrected_img)
  
  put_filter(face_img, filter_img, reference_img)
   