import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model weights
model = DL_model(input_shape=(224, 224, 3), n_classes=6)
model.load_weights('vgg16_model_best_weights.h5')

# Load and preprocess the image
img_path = '/Users/bhuvaneshmarineni/ODU subject related/Computer Vision/apple.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = keras.applications.vgg16.preprocess_input(x)

# Make predictions on the input image
preds = model.predict(x)
classes = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
result = classes[np.argmax(preds)]

# Check if the fruit is fresh or rotten
if 'fresh' in result:
    print('The fruit is fresh')
else:
    print('The fruit is rotten')
