import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Chargement de l'ensemble de données (ex : MNIST)
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images - 127.5) / 127.5  # Normalisation des pixels entre -1 et 1

# Paramètres
random_dim = 100
epochs = 10000
batch_size = 64

# Création du générateur
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=random_dim, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(784, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

# Création du discriminateur
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Création du GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Compilation des modèles
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# Entraînement du GAN
for epoch in range(epochs):
    noise = np.random.normal(0, 1, size=[batch_size, random_dim])
    generated_images = generator.predict(noise)
    image_batch = train_images[np.random.randint(0, train_images.shape[0], size=batch_size)]

    # Ajout d'une dimension pour les images en niveaux de gris
    image_batch = np.expand_dims(image_batch, axis=-1)

    X = np.concatenate([image_batch, generated_images])
    y_dis = np.zeros(2 * batch_size)
    y_dis[:batch_size] = 0.9

    discriminator.trainable = True
    d_loss = discriminator.train_on_batch(X, y_dis)

    noise = np.random.normal(0, 1, size=[batch_size, random_dim])
    y_gen = np.ones(batch_size)
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noise, y_gen)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")

        # Affichage des images générées
        if epoch % 1000 == 0:
            generated_images = generator.predict(np.random.normal(0, 1, size=[16, random_dim]))
            generated_images = generated_images * 0.5 + 0.5  # Remettre à l'échelle entre 0 et 1
            fig, axs = plt.subplots(4, 4)
            count = 0
            for i in range(4):
                for j in range(4):
                    axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
                    axs[i, j].axis('off')
                    count += 1
            plt.show()