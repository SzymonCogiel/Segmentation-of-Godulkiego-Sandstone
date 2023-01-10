import cv2
import pickle
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.ndimage import median_filter
from skimage.morphology import remove_small_objects

# set seeds
seed(42)

# shape
width = 968
height = 1292


class Sandstone:

    feature_extractor_mika = None
    feature_extractor_kwarc = None
    feature_extractor_glauk = None
    mika_model = None
    kwarc_model = None
    glau_model = None
    test_img = None

    def __init__(self, path):
        self.load_models()
        self.load_img(path=path)

    @staticmethod
    def load_img(path: str):
        test_img = cv2.imread(path, cv2.IMREAD_COLOR)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        test_img = np.expand_dims(test_img, axis=0)
        return test_img

    # load created before models
    def load_models(self):
        self.feature_extractor_mika = load_model("data/models/extractor_mika.h5")
        self.feature_extractor_kwarc = load_model("data/models/extractor_kwarc.h5")
        self.feature_extractor_glauk = load_model("data/models/extractor_glauk.h5")
        self.mika_model = pickle.load(open('data/models/RF_model_mika.sav', 'rb'))
        self.kwarc_model = pickle.load(open('data/models/RF_model_kwarc.sav', 'rb'))
        self.glau_model = pickle.load(open('data/models/RF_model_glau.sav', 'rb'))

    # extract features
    def __feature_extractors(self, feature_extractor):
        X_test_feature = feature_extractor.predict(self.test_img)
        X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])
        return X_test_feature

    # predict image by base model
    def __predict_mineral(self, model, feature_extractor):
        predicted_img = model.predict(self.__feature_extractors(feature_extractor))
        predicted_img = predicted_img.reshape((968, 1292))
        return predicted_img

    # additional morpholog operations to improve masks
    def __morphology_operations(self, img, min_size, kernel):
        img = np.array(img, dtype=bool)
        img = remove_small_objects(img, min_size=min_size)
        img.dtype = 'uint8'
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return img

    # predict minelars
    def predict(self):

        mika_img = self.__predict_mineral(self.mika_model, self.feature_extractor_mika)
        glau_img = self.__predict_mineral(self.glau_model, self.feature_extractor_glauk)
        kwarc_img = self.__predict_mineral(self.kwarc_model, self.feature_extractor_kwarc)

        mika_img = self.__morphology_operations(mika_img, 120, np.ones((7, 7), np.uint8))
        glau_img = self.__morphology_operations(glau_img, 400, np.ones((9, 9), np.uint8))








