from pytube import YouTube

yt = YouTube('https://www.youtube.com/watch?v=A5jciTbgOxY')

print(yt.title)

stream = yt.streams.get_by_itag('137')
stream.download()
print(stream)

'''
stream = yt.streams.filter(only_audio=True).get_by_itag('140')
print(stream)
'''

'''
# itag = '137'
import cv2

jardy_openpose = cv2.VideoCapture()

fps = jardy_openpose.get(cv2.cv.CV_CAP_PROP_FPS)
print(fps)

'''
