import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build a simple CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', name='conv2'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', name='conv3'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define a callback to record training history
class PlotFeatures(tf.keras.callbacks.Callback):
    def __init__(self, sample_image):
        self.sample_image = sample_image

    def on_epoch_end(self, epoch, logs=None):
        layer_names = ['conv1', 'conv2', 'conv3']
        intermediate_layer_models = [models.Model(inputs=model.input,
                                                  outputs=model.get_layer(name).output)
                                     for name in layer_names]

        feature_maps = [model.predict(self.sample_image.reshape(1, 28, 28, 1)) for model in intermediate_layer_models]

        # Plot feature maps
        for i, feature_map in enumerate(feature_maps):
            plt.figure(figsize=(15, 3))
            for j in range(feature_map.shape[-1]):
                plt.subplot(1, feature_map.shape[-1], j + 1)
                plt.imshow(feature_map[0, :, :, j], cmap='viridis')
                plt.title(f'Layer {layer_names[i]} - Channel {j}')
                plt.axis('off')
            plt.show()

# Train the model with the callback
sample_image = test_images[0]
model.fit(train_images, train_labels, epochs=5, batch_size=64, callbacks=[PlotFeatures(sample_image)])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
