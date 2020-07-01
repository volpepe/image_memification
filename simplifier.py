import cv2
import numpy as np
from PIL import Image
import argparse
from sklearn.cluster import MiniBatchKMeans

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, 
                        required=True, help="path (or URL) to image")
    parser.add_argument("-c", "--colors", type=int,
                        default=5, help="number of colors")
    args = parser.parse_args()
    return vars(args)

def image_PIL_to_opencv(pil_image):
    img = np.asarray(pil_image)
    result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return result
    
def get_image(URI):
    try:
        response = requests.get(URI)
        img = Image.open(BytesIO(response.content))
    except requests.ConnectionError as exception:
        img = Image.open(URI)
    return img

def simplify_image_palette(img_path, colors):
    image = get_image(img_path)
    result = image.convert('P', palette=Image.ADAPTIVE, colors=colors).convert('RGB')
    result = image_PIL_to_opencv(result)
    result.show()
    return result

def simplify_image_kmeans(image, colors):
    """ https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/ """
    (h, w) = image.shape[:2]
    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = colors)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    return quant

def main():
    args = parse_args()
    result = simplify_image_kmeans(get_image(args["image"]), args["colors"])
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()