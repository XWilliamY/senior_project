import cv2

'''
jardy_openpose = cv2.VideoCapture('../jardy_output_1.mp4')

fps = jardy_openpose.get(cv2.CAP_PROP_FPS)

print(fps)
'''
jardy_pytube = cv2.VideoCapture('itzy.mp4')
fps = jardy_pytube.get(cv2.CAP_PROP_FPS)
print(fps)
