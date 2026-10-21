
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

# Load Fashion MNIST dataset
(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]


# Create a simple convolutional Autoencoder
def build_autoencoder():
    inputs = keras.Input(shape=(28, 28, 1))

    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2, padding="same")(x)

    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D(2, padding="same")(x)

    x = layers.Conv2D(8, 3, activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D(2)(x)

    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)

    outputs = layers.Conv2D(
        1, 3, activation="sigmoid", padding="same"
    )(x)

    return keras.Model(inputs, outputs)


# Display images
def display_images(original, reconstructed, title):
    fig, axes = plt.subplots(1, len(original) * 2, figsize=(12, 3))

    for i in range(len(original)):
        axes[2 * i].imshow(original[i].squeeze(), cmap="gray")
        axes[2 * i].set_title("Original")
        axes[2 * i].axis("off")

        axes[2 * i + 1].imshow(
            reconstructed[i].squeeze(), cmap="gray"
        )
        axes[2 * i + 1].set_title("Reconstructed")
        axes[2 * i + 1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ==================================================
# TASK 1: Autoencoder using MSE loss, 5 epochs
# ==================================================
print("TASK 1: Training Autoencoder with MSE")

mse_model = build_autoencoder()
mse_model.compile(optimizer="adam", loss="mse")

mse_model.fit(
    x_train, x_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, x_test)
)

prediction = mse_model.predict(x_test[:1], verbose=0)
display_images(x_test[:1], prediction, "Task 1: MSE Autoencoder")


# ==================================================
# TASK 2: Autoencoder using Binary Crossentropy
# ==================================================
print("TASK 2: Training Autoencoder with BCE")

bce_model = build_autoencoder()
bce_model.compile(
    loss="binary_crossentropy",
    optimizer="adam"
)

bce_model.fit(
    x_train, x_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, x_test)
)

indices = [0, 1, 2]
original = x_test[indices]

mse_predictions = mse_model.predict(original, verbose=0)
bce_predictions = bce_model.predict(original, verbose=0)

fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for i in range(3):
    axes[i, 0].imshow(original[i].squeeze(), cmap="gray")
    axes[i, 0].set_title("Original")

    axes[i, 1].imshow(mse_predictions[i].squeeze(), cmap="gray")
    axes[i, 1].set_title("MSE")

    axes[i, 2].imshow(bce_predictions[i].squeeze(), cmap="gray")
    axes[i, 2].set_title("BCE")

    for j in range(3):
        axes[i, j].axis("off")

plt.suptitle("Task 2: MSE vs BCE Comparison")
plt.tight_layout()
plt.show()


# ==================================================
# TASK 3: Denoising Autoencoder
# Train noisy images as input and clean images as targets
# ==================================================
print("TASK 3: Training Denoising Autoencoder")

noise_train = np.random.normal(
    0, 0.3, x_train.shape
).astype("float32")

noise_test = np.random.normal(
    0, 0.3, x_test.shape
).astype("float32")

x_train_noisy = np.clip(x_train + noise_train, 0, 1)
x_test_noisy = np.clip(x_test + noise_test, 0, 1)

denoise_model = build_autoencoder()
denoise_model.compile(optimizer="adam", loss="mse")

denoise_model.fit(
    x_train_noisy, x_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test_noisy, x_test)
)

noisy_samples = x_test_noisy[:10]
denoised_samples = denoise_model.predict(noisy_samples, verbose=0)

fig, axes = plt.subplots(10, 3, figsize=(8, 20))

for i in range(10):
    axes[i, 0].imshow(x_test[i].squeeze(), cmap="gray")
    axes[i, 0].set_title("Original" if i == 0 else "")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(noisy_samples[i].squeeze(), cmap="gray")
    axes[i, 1].set_title("Noisy" if i == 0 else "")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(denoised_samples[i].squeeze(), cmap="gray")
    axes[i, 2].set_title("Denoised" if i == 0 else "")
    axes[i, 2].axis("off")

plt.suptitle("Task 3: Denoising Results")
plt.tight_layout()
plt.show()


# ==================================================
# TASK 4: Variational Autoencoder (VAE)
# ==================================================
print("TASK 4: Training Variational Autoencoder")

latent_dim = 2


# Sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# VAE Encoder
encoder_input = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(
    32, 3, strides=2, padding="same", activation="relu"
)(encoder_input)

x = layers.Conv2D(
    64, 3, strides=2, padding="same", activation="relu"
)(x)

x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(
    encoder_input, [z_mean, z_log_var, z], name="encoder"
)


# VAE Decoder
decoder_input = keras.Input(shape=(latent_dim,))

x = layers.Dense(7 * 7 * 64, activation="relu")(decoder_input)
x = layers.Reshape((7, 7, 64))(x)

x = layers.Conv2DTranspose(
    64, 3, strides=2, padding="same", activation="relu"
)(x)

x = layers.Conv2DTranspose(
    32, 3, strides=2, padding="same", activation="relu"
)(x)

decoder_output = layers.Conv2D(
    1, 3, padding="same", activation="sigmoid"
)(x)

decoder = keras.Model(
    decoder_input, decoder_output, name="decoder"
)


# VAE model with reconstruction and KL-divergence loss
class VAE(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.total_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.total_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)

            bce = keras.backend.binary_crossentropy(
                data, reconstruction
            )

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(bce, axis=(1, 2, 3))
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + log_var - tf.square(mean) - tf.exp(log_var),
                    axis=1
                )
            )

            total_loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(
            total_loss, self.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_weights)
        )

        self.total_tracker.update_state(total_loss)

        return {"loss": self.total_tracker.result()}


vae = VAE(encoder, decoder)
vae.compile(optimizer="adam")

vae.fit(
    x_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, None)
)


# Generate five new images from random latent vectors
random_points = np.random.normal(
    size=(5, latent_dim)
).astype("float32")

generated_images = decoder.predict(random_points, verbose=0)

fig, axes = plt.subplots(1, 5, figsize=(12, 3))

for i in range(5):
    axes[i].imshow(generated_images[i].squeeze(), cmap="gray")
    axes[i].set_title(f"Generated {i + 1}")
    axes[i].axis("off")

plt.suptitle("Task 4: Images Generated by VAE")
plt.tight_layout()
plt.show()


# ==================================================
# TASK 5: KL-Divergence explanation
# ==================================================
print("""
KL-Divergence Loss in a VAE:

KL-divergence measures the difference between the latent
distribution learned by the encoder and a standard normal
distribution.

It encourages the latent space to remain smooth and organized.
This allows us to sample random latent vectors and generate
new images using the decoder.

Without KL-divergence loss, the latent space may become
irregular, and random samples may generate poor images.

Reconstruction loss helps reconstruct the input image, while
KL-divergence loss organizes the latent space for generation.
""")
