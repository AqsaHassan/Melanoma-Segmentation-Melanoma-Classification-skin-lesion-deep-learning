import tensorflow as tf
import numpy as np
import cv2
import os
import sklearn.preprocessing
import glob



def preprocessing(img):
    size = (512, 384)
    img2 = cv2.resize(img, size)
    img2 = np.clip(img2 - np.median(img2)+127, 0, 255)
    img2 = img2.astype(np.float32)

    img2 = img2/255.
    return img2


def get_predict(model, img):
    processed_img = preprocessing(img)
    processed_img = np.expand_dims(processed_img, axis=0)
    out_put = model.predict(processed_img)
    out_put = out_put[..., -1]
    out_put = out_put[0]
    out_put = sklearn.preprocessing.binarize(out_put, threshold=0.5)
    new_image = processed_img*np.reshape(out_put, (out_put.shape+(1,)))


    return out_put


file_path_model = 'model.h5'
trained_model = tf.keras.models.load_model(file_path_model, compile=False)
size = (512, 384)


def get_masked_image(image_path):

    img2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = get_predict(trained_model, img2)

    resized_image = cv2.resize(img2, size)
    new_image = resized_image*np.reshape(mask, (mask.shape+(1,)))

    return mask, new_image



####--------------test code

images_folder = './../test/images/*'
file_img = glob.glob(images_folder)

for index, name in enumerate(file_img):

    mask, new_image = get_masked_image(name)

    os.makedirs("./output", exist_ok=True)
    cv2.imwrite("./output/img_{0}.png".format(index), new_image)
