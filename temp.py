from feature import NPDFeature
from PIL import Image
import numpy as np
import pickle



def save(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

img = Image.open('./datasets/original/face/face_000.jpg').convert('L').resize((24,24))
f = NPDFeature(np.array(img))
feature = f.extract()
save(feature,'./feature')

ff = load('./feature')

