import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc

width = 1280
height = 720
FPS = 30
seconds = 5

fourcc = VideoWriter_fourcc(*'MP42') #codec
video = VideoWriter('funnie_image_generator/noise.avi', fourcc, float(FPS), (width, height)) #writer

#we use a loop to write a random array in each frame
for frame in range(FPS*seconds):

    video_frame = np.random.randint(0, 256,
                            (height, width, 3),
                            dtype=np.uint8)
    
    video.write(video_frame)

video.release()