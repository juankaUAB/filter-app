from functions import *

def get_filter_mask(img, filtre):
    mask1 = cv2.cvtColor(cv2.subtract(img, filtre), cv2.COLOR_BGR2GRAY)
    a = np.uint8((mask1) > 100)
    
    kernel_d = np.ones((2,2), np.uint8)
    kernel_e = np.ones((4,4), np.uint8)
    
    mask = cv2.dilate(a, kernel_d, iterations = 1)
    mask = cv2.erode(mask, kernel_e, iterations = 1)
        
    return mask

def filter_apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


def get_filter(img, plantilla):
    mask = get_filter_mask(plantilla, img)
    f = filter_apply_mask(img, mask)
    
    return f

'''
plantilla = cv2.imread("../imgs/plantilla.jpg", cv2.IMREAD_COLOR)

dibuix = cv2.imread("../imgs/foto_gt.jpg", cv2.IMREAD_COLOR)
#dibuix = cv2.imread("../imgs/foto_real.jpg", cv2.IMREAD_COLOR)

corrected_img, h = alignImages(dibuix, plantilla)

cv2.imshow('hola', corrected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

f = get_filter(corrected_img, plantilla)
cv2.imwrite('objetos_filtro.jpg', f)

cv2.imshow('hola', f)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
 