import cv2
import numpy as np

MAX_FEATURES = 200
GOOD_MATCH_PERCENT = 0.15

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


def get_filter(img, ref):
    return cv2.subtract(ref, img)

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
    
    
    
    
    
    
    
