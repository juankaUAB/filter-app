import cv2
import dlib
import numpy
import copy
from filtre import *

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

  return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

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
  im = numpy.ones(im.shape[:2], dtype=numpy.float64)
  
  '''
  for group in OVERLAY_POINTS:
    draw_convex_hull(im,
                     landmarks[group],
                     color=1)
    
    '''
  im = numpy.array([im, im, im]).transpose((1, 2, 0))

  im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT),  0) > 0) * 1.0
  im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
  return im

def transformation_from_points(points1, points2):

# Resolver el problema de procrustes restando los centroides, escalando por el
# Desviación estándar, y luego usar el SVD para calcular la rotación. Ver
# Lo siguiente para más detalles:
# https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

  points1 = points1.astype(numpy.float64)
  points2 = points2.astype(numpy.float64)

  c1 = numpy.mean(points1, axis=0)
  c2 = numpy.mean(points2, axis=0)
  points1 -= c1
  points2 -= c2

  s1 = numpy.std(points1)
  s2 = numpy.std(points2)
  points1 /= s1
  points2 /= s2

  U, S, Vt = numpy.linalg.svd(points1.T * points2)

# El R que buscamos es, de hecho, la transposición de la dada por U * Vt. Esto
# Es porque la formulación anterior asume que la matriz va a la derecha
# (Con vectores de fila) donde como nuestra solución requiere que la matriz esté en la
# Izquierda (con vectores columna)
  R = (U * Vt).T

  return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                     c2.T - (s2 / s1) * R * c1.T)),
                       numpy.matrix([0., 0., 1.])])

def read_im_and_landmarks(fname):
  im = cv2.imread(fname, cv2.IMREAD_COLOR)
  im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
  im.shape[0] * SCALE_FACTOR))
  s = get_landmarks(im)

  return im, s

def warp_im(im, M, dshape):
  output_im = numpy.zeros(dshape, dtype=im.dtype)
  cv2.warpAffine(im,
                 M[:2],
                 (dshape[1], dshape[0]),
                 dst=output_im,
                 borderMode=cv2.BORDER_TRANSPARENT,
                 flags=cv2.WARP_INVERSE_MAP)
  return output_im

def correct_colours(im1, im2, landmarks1):
  blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
  blur_amount = int(blur_amount)
  if blur_amount % 2 == 0:
    blur_amount += 1
  im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
  im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

# evitamos dividir por 0
  im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

  return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                              im2_blur.astype(numpy.float64))


#aqui se añaden las imagenes que queremos usar
im1, landmarks1 = read_im_and_landmarks("../imgs/fotojuanka.jpg") 
im2, landmarks2 = read_im_and_landmarks("../imgs/foto_gt.jpg") #faltaria aqui alinear el filtro
plantilla = cv2.imread("../imgs/plantilla.jpg", cv2.IMREAD_COLOR)

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
im1_copy[numpy.nonzero(warped_im2)] = warped_im2[numpy.nonzero(warped_im2)]

#imagen de salida
cv2.imwrite('output.jpg', im1_copy)