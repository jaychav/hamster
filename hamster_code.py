# Libraries!
import numpy as np
import cv2 as cv
from PIL import Image
import os

# cascPath = "./haarcascade_eye.xml"
cascPath = "./haarcascade_frontalface_default.xml"
# cascPath = "./haarcascade_frontalface_alt_tree.xml"
# cascPath = "./haarcascade_frontalface_alt.xml"
# cascPath = "./haarcascade_profileface_newer.xml"

faceCascade = cv.CascadeClassifier(cascPath)

def overlay_image_alpha(img, img_overlay_in, x, y, w, output_file):
  """
  Overlay `img_overlay` onto `img` at (x,y) and blend using `alpha_mask`.
  `alpha_mask` must have the same H and W as `img_overlay` and values in range [0,1]
  """
  imgNew = np.array(Image.open(img))
  img_copy = imgNew[:,:,:3].copy()
  img_overlay_in_resized = resize_img(img_overlay_in, w, "resized_img.png")
  img_overlay_rgba = np.array(Image.open(img_overlay_in_resized))
  alpha_mask = img_overlay_rgba[:,:,:3] / 255.0
  
  img_overlay = img_overlay_rgba[:,:,:3]
  
  # image rangers:
  y1,y2 = max(0,y), min(img_copy.shape[0], y + img_overlay.shape[0])
  x1,x2 = max(0,x), min(img_copy.shape[1], y + img_overlay.shape[1])
  
  # overlay ranges
  y1o, y2o = max(0, -y), min(img_overlay.shape[0], y + img_copy.shape[0] - y)
  x1o, x2o = max(0, -x), min(img_overlay.shape[1], y + img_copy.shape[1] - x)
  
  # exit if we don't have anything to do
  if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
    return
  
  # if not, blend overlay within determined ranges
  img_crop = img_copy[y1:y2, x1:x2]
  img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
  alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
  alpha_inv = 1.0 - alpha
  
  img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
  
  Image.fromarray(img_copy).save(output_file)
  os.remove("resized_img.png")
  
def resize_img(img_in, w, img_out):
  img = Image.open(img_in)
  
  basewidth = w
  wpercent = (basewidth/float(img.size[0]))
  hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((basewidth, hsize), Image.ANTIALIAS)
  img.save(img_out)
  return img_out

# resize_img("hamster1.png", 100, "hamster1resized.png")

def hamsterize_img(img_in):
  image_faces_rgb = cv.imread(img_in)
  image_faces_gray = cv.cvtColor(image_faces_rgb, cv.COLOR_RGB2GRAY)
  faces = faceCascade.detectMultiScale(
    images_faces_gray,  ## this is the input image
    scaleFactor = 1.05, ## this is the scale-resolution for detecting faces ##FIXME
    minNeighbors = 1,   ## this is how many nearby-detections are needed to OK a face ##FIXME
    minSize = (10,10),  ## this is the minimum size for a face ##FIXME
    flags = cv.CASCADE_SCALE_IMAGE, # (standard)
  )
  for i, face in enumerate(faces):
    x,y,w,h = face
  img = Image.open(img_in)
  output_file = img_in[:-4] + "_hamsterized" + img_in[-4:]
  img_overlay = "hamster1.png"
  img.save(output_file)
  input_file = output_file
  
  for (x,y,w,h) in faces:
    overlay_image_alpha(input_file, img_overlay, x, y, w, output_file)
    
  ## show the image:
  ## Image.open(output_file).show()
  
hamsterize_img("spaghetti_night.png")
