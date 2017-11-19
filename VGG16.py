from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np
import os

# Save and load model
PATH = "../input/VGG16.h5"
if not os.path.isfile(PATH):
    model = VGG16(weights="imagenet", include_top=True)
    model.save(PATH)
else:
    model = load_model(PATH)

# Print model summary
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output_shape)


PATH = "data/images.jpeg"
x = image.load_img(PATH)
x = image.img_to_array(x)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

print(x)

