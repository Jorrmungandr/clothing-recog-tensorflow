import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from network import model, test_images, test_labels

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot',
]

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.colorbar()
    plt.xlabel('Real: {0}\n Predict: {1}'.format(
        class_names[test_labels[i]],
        class_names[np.argmax(predictions[i])]
    ))
plt.show()
