import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np
import argparse
from simplifier import simplify_image_kmeans
import random
import requests

FPS = 500

def show_image(image, name="Image"):
    # display the image to our screen -- we will need to click the window
    # open by OpenCV and press a key on our keyboard to continue execution
    cv2.imshow(name, image)
    cv2.waitKey(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, 
                        required=True, help="path (or URL) to image")
    parser.add_argument("-c", "--colors", type=int,
                        default=10, help="number of colors")
    parser.add_argument("-fn", "--filename", type=str,
                        default="test", help="name of output file")
    args = parser.parse_args()
    return vars(args)

def resize(image, dim, keep_ratio=False):
    #resize an image without considering the aspect ratio
    h, w, d = image.shape
    if not keep_ratio:
        resized = cv2.resize(image, (dim, dim))
    else:
        #calculate the aspect ratio
        # h:w = x:dim --> x = dim*h/w
        new_height = dim * h / w
        resized = cv2.resize(image, (dim, int(new_height)))
    return resized

def main_loop(video_writer, image, w, h):
    print("Image has size {}x{}".format(w, h))
    n_frames = 0
    canvas = np.zeros((h, w, 3), np.uint8)
    
    #choose a random coordinate on the image
    hs, ws = list(range(h-1)), list(range(w-1))
    random.shuffle(hs)
    random.shuffle(ws)

#    hs_c = [hs[10*i:10*(i+1)] for i in range(len(hs)//10)]
#    if len(hs) % 10 == 0:
#        hs_c.extend(hs[-(len(hs)%10):])

#    for y_list in hs_c:
    for x in ws:
        for y in hs:
            if tuple(canvas[y,x]) == (0,0,0):
                #some output
                if n_frames % 1000 == 0:
                    print("Drawn {} frames".format(n_frames))

                #get the color
                color = tuple(image[y, x]) #(B, G, R)

                #follow line to extreme points
                sx = ex = x
                while(sx > 0 and tuple(image[y, sx - 1]) == color):
                    sx -= 1
                while(ex < w-1 and tuple(image[y, ex + 1]) == color):
                    ex += 1

                #extend line checklist
#               checklist.extend(list(range(sx, ex)))

                #update canvas
                cv2.line(canvas, (sx, y), (ex, y), tuple(int(x) for x in color), 2)

                #draw updated canvas
                video_writer.write(np.asarray(canvas))
                n_frames += 1

    print("Video ended with {} frames".format(n_frames))


def main():
    args = parse_args()
    # Treat image path as URL: if it's not, just open it
    try:
        resp = requests.get(args["image"])
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except requests.ConnectionError as exception:
        image = cv2.imread(args["image"])    
    image = resize(image, 480, keep_ratio=True)
    image = simplify_image_kmeans(image, args["colors"])
    h, w = image.shape[0], image.shape[1]
    fourcc = VideoWriter_fourcc(*'MP42') #codec
    writer = VideoWriter('{}.avi'.format(args["filename"]), 
                    fourcc, float(FPS), (w, h)) #writer
    main_loop(writer, image, w, h)
    writer.release()    

if __name__ == "__main__":
    main()