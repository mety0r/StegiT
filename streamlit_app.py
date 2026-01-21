import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Function to build the generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(28 * 28 * 1, activation='tanh')) 
    model.add(tf.keras.layers.Reshape((28, 28, 1)))  
    return model

# Function to build the discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  
    return model

# Function to train the GAN
def train_gan(generator, discriminator, gan, epochs=1000, batch_size=128, sample_interval=100):
    
    for epoch in range(epochs):
        
        if epoch % sample_interval == 0:
            # Generate images and save/display them
            noise = np.random.normal(0, 1, (1, 100))
            generated_image = generator.predict(noise)
            save_generated_image(generated_image, epoch) 

# Function to save generated images
def save_generated_image(image, epoch):
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')
    image = tf.squeeze(image, axis=0)
    image = (image + 1) / 2.0  
    image = tf.image.convert_image_dtype(image, tf.uint8)
    tf.keras.preprocessing.image.save_img(f'generated_images/generated_{epoch}.png', image)

# Initialize Streamlit app
def main():
    st.title("GAN Image Generation with Streamlit")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(28, 28))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to a numpy array and preprocess it
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Build the generator and discriminator
        generator = build_generator()
        discriminator = build_discriminator()

        # Compile the discriminator
        discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Compile the GAN
        discriminator.trainable = False
        gan_input = tf.keras.layers.Input(shape=(100,))
        generated_image = generator(gan_input)
        gan_output = discriminator(generated_image)
        gan = tf.keras.models.Model(gan_input, gan_output)
        gan.compile(optimizer='adam', loss='binary_crossentropy')

        # Train the GAN (you may adjust epochs and batch_size)
        train_gan(generator, discriminator, gan, epochs=1000, batch_size=128, sample_interval=100)

        # Display generated images
        st.subheader("Generated Images")
        generated_image_files = os.listdir('generated_images')
        for image_file in generated_image_files:
            image_path = os.path.join('generated_images', image_file)
            st.image(image_path, caption=f"Generated Image {image_file}", use_column_width=True)

if __name__ == '__main__':
    main()

