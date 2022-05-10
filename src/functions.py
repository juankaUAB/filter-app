import cv2
import numpy as np
import dlib
import copy

MAX_FEATURES = 750
GOOD_MATCH_PERCENT = 0.15

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Puntos utilizados para alinear las imágenes.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Puntos de la segunda imagen para superponer en la primera. El casco convexo de cada
# Elemento será superpuesto.
OVERLAY_POINTS = [
LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
NOSE_POINTS + MOUTH_POINTS,
]

COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def alignImages(im1, im2):

  # Convert images to grayscale
  gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

  # Match features
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches = sorted(matches,key=lambda x: x.distance, reverse=False)

  # Select best matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw lines corresponding to the matches
  img_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("../imgs/matches.jpg", img_matches)

  # Extract location of selected matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Apply transformations to align the image
  height, width, channels = im2.shape
  align_img = cv2.warpPerspective(im1, h, (width, height))

  return align_img, h


#modificacion del juanka: usamos threshold adaptativo para que no afecte la iluminacion
def binarize(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(img, 5)
    dst2 = np.logical_not(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
    
    return dst2

#funcion del juanka
def get_objects(reference_img, align_img):
    applied_mask = np.zeros((align_img.shape[0],align_img.shape[1],align_img.shape[2]),dtype=np.uint8)
    applied_mask[:,:,0] =  align_img[:,:,0] * reference_img
    applied_mask[:,:,1] =  align_img[:,:,1] * reference_img
    applied_mask[:,:,2] =  align_img[:,:,2] * reference_img
    #applied_mask[applied_mask == 0] = 255
    return applied_mask

def detect_face(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces
    
    '''for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    return img'''

def take_picture():
    cam = cv2.VideoCapture(0)
    while (cam.isOpened()):
        ret, frame = cam.read()
        cv2.imshow('Fer foto amb la tecla f',frame)
        if (cv2.waitKey(1) == ord('f')):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
    return frame
    
def apply_filter(img, filtre, faces):
    for (x, y, w, h) in faces:
        h = h + 20
        nou_filtre = cv2.resize(filtre, (w, h))
        img[x:x+h, y:y+w, :] = filter2face(img[x:x+h, y:y+w, :], nou_filtre)
    
    return img
    
def filter2face(img, filtre):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if np.all(filtre[row,col,:] > [20, 20, 20]):
                img[row, col, :] = filtre[row, col, :]
                
    return img

class TooManyFaces(Exception):
  pass

class NoFaces(Exception):
  pass

def get_landmarks(im):
  rects = detector(im, 1)

  if len(rects) > 1:
    raise TooManyFaces
  if len(rects) == 0:
    raise NoFaces

  return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
  im = im.copy()
  for idx, point in enumerate(landmarks):
    pos = (point[0, 0], point[0, 1])
    cv2.putText(im, str(idx), pos,
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                fontScale=0.4,
                color=(0, 0, 255))
    cv2.circle(im, pos, 3, color=(0, 255, 255))
  return im

def draw_convex_hull(im, points, color):
  points = cv2.convexHull(points)
  cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
  im = np.ones(im.shape[:2], dtype=np.float64)
  
  '''
  for group in OVERLAY_POINTS:
    draw_convex_hull(im,
                     landmarks[group],
                     color=1)
    
    '''
  im = np.array([im, im, im]).transpose((1, 2, 0))

  im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT),  0) > 0) * 1.0
  im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
  return im

def transformation_from_points(points1, points2):

# Resolver el problema de procrustes restando los centroides, escalando por el
# Desviación estándar, y luego usar el SVD para calcular la rotación. Ver
# Lo siguiente para más detalles:
# https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

  points1 = points1.astype(np.float64)
  points2 = points2.astype(np.float64)

  c1 = np.mean(points1, axis=0)
  c2 = np.mean(points2, axis=0)
  points1 -= c1
  points2 -= c2

  s1 = np.std(points1)
  s2 = np.std(points2)
  points1 /= s1
  points2 /= s2

  U, S, Vt = np.linalg.svd(points1.T * points2)

# El R que buscamos es, de hecho, la transposición de la dada por U * Vt. Esto
# Es porque la formulación anterior asume que la matriz va a la derecha
# (Con vectores de fila) donde como nuestra solución requiere que la matriz esté en la
# Izquierda (con vectores columna)
  R = (U * Vt).T

  return np.vstack([np.hstack(((s2 / s1) * R,
                                     c2.T - (s2 / s1) * R * c1.T)),
                       np.matrix([0., 0., 1.])])

def read_im_and_landmarks(im):
  im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
  im.shape[0] * SCALE_FACTOR))
  s = get_landmarks(im)

  return im, s

def warp_im(im, M, dshape):
  output_im = np.zeros(dshape, dtype=im.dtype)
  cv2.warpAffine(im,
                 M[:2],
                 (dshape[1], dshape[0]),
                 dst=output_im,
                 borderMode=cv2.BORDER_TRANSPARENT,
                 flags=cv2.WARP_INVERSE_MAP)
  return output_im

def put_filter(im1, im2, plantilla): #im1 foto de la cara, im2 foto del filtro
    im1, landmarks1 = read_im_and_landmarks(im1) 
    im2, landmarks2 = read_im_and_landmarks(im2) #faltaria aqui alinear el filtro

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])

    im2_copy = copy.deepcopy(im2)
    im2_copy = get_filter(im2_copy, plantilla)
    mask = get_face_mask(im2_copy, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    #combined_mask = get_face_mask(im1, landmarks1)

    warped_im2 = warp_im(im2_copy, M, im1.shape)
    #warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

    warped_im2 = cv2.normalize(warped_im2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #warped_im2: plantilla alineada
    #im1: foto para ponerle la plantilla

    im1_copy = copy.deepcopy(im1)
    im1_copy[np.nonzero(warped_im2)] = warped_im2[np.nonzero(warped_im2)]

    #imagen de salida
    cv2.imwrite('../imgs/output.jpg', im1_copy)
    
    
    
    
    
