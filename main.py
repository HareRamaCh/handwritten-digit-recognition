import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build and train the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

model.save('mnist_model.h5')  # Save the trained model

# GUI for Handwritten Digit Recognition
class DigitRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Recognition")

        self.canvas = Canvas(master, width=280, height=280, bg="white", cursor="cross")
        self.canvas.grid(row=0, column=0, padx=10, pady=10, columnspan=3)

        self.label = Label(master, text="Draw a digit and click 'Predict'")
        self.label.grid(row=1, column=0, columnspan=3)

        self.predict_button = Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=2, column=1)

        self.clear_button = Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=2)

        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.model = load_model('mnist_model.h5')

    def predict_digit(self):
        img_array = np.array(self.image)
        img_array = self.preprocess_image(img_array)
        probabilities = self.model.predict(img_array)[0]
        prediction = np.argmax(probabilities)
        self.label.config(text=f"Predicted Digit: {prediction}")

    def preprocess_image(self, img_array):
        # Convert to PIL Image
        img_pil = Image.fromarray(img_array)

        # Resize the image
        img_pil = img_pil.resize((28, 28))

        # Convert back to NumPy array
        img_array_resized = np.array(img_pil)

        # Ensure grayscale
        if len(img_array_resized.shape) == 3:
            img_array_resized = img_array_resized.mean(axis=-1)

        img_array_resized = img_array_resized.reshape((1, 28, 28, 1)).astype('float32') / 255
        return img_array_resized

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
        self.draw.line([x1, y1, x2, y2], fill="black", width=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    app.canvas.bind("<B1-Motion>", app.paint)
    root.mainloop()
