import cv2
import numpy as np
import glob
 
img_array = []
filenames = sorted(glob.glob('/Users/will.i.liam/Desktop/final_project/phoan/images/*.jpg'))

count = 0
for filename in filenames:
    if count == 1800:
        break
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    count += 1
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
