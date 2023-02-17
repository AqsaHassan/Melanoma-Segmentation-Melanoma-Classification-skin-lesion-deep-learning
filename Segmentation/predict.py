import tensorflow as tf
import numpy as np
from data import get_predict
import cv2
import os
import sklearn.preprocessing
import glob
file_path_model = './../model.h5'
trained_model = tf.keras.models.load_model(file_path_model, compile=False)
size = (512, 384)
images_folder = './test/images/*'
file_img = glob.glob(images_folder)
abc = []
for index, name in enumerate(file_img):
    print(name)
#    name = images_folder + '/'+i
    img2 = cv2.imread(name, cv2.IMREAD_COLOR)
    out_put = get_predict(trained_model, img2)
    out_put = out_put.astype(np.uint8)
    out_put = np.stack([out_put, out_put, out_put], axis=-1)
    out_put2 = np.concatenate([cv2.resize(img2, size), out_put], axis=1)
    abc.append(out_put2)
    os.makedirs("./output", exist_ok=True)
    cv2.imwrite("./output/img_{0}.png".format(index), out_put2)


abcd = np.concatenate(abc,axis=0)
cv2.cvtColor(abcd,cv2.COLOR_BGR2RGB)
