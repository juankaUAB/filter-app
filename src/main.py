from functions import *

if __name__ == '__main__':

  # Read reference image
  reference_img = cv2.imread("../imgs/plantilla.jpg", cv2.IMREAD_COLOR)

  # Read image to be aligned
  align_img = cv2.imread("../imgs/foto_gt.jpg", cv2.IMREAD_COLOR)

  #Calculate the homography and store it in h with the corrected img in corrected_img
  corrected_img, h = alignImages(align_img, reference_img)

  # Write aligned image to disk.
  #cv2.imwrite("../imgs/aligned.jpg", corrected_img)

  b = binarize(corrected_img)
  klk = get_objects(b, corrected_img)
    
  cv2.imshow('hola', corrected_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # Write aligned image to disk.
  cv2.imshow('hola', klk)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  foto = take_picture()
  faces = detect_face(foto)
  
  cv2.imshow('carica', apply_filter(foto, klk, faces))
  cv2.waitKey(0)
  cv2.destroyAllWindows()
   