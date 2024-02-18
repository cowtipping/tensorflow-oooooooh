"""Visualise the model's predictions."""

import json
from tensorflow.keras.models import load_model
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from main import x_test, y_test

# Load the model
model = load_model("mymodel.keras")

# Make predictions on the test set
predictions = model.predict(x_test)

# Convert predictions to a list
predictions_list = predictions.tolist()

# Save to a JSON file
with open("predictions.json", "w") as f:
    json.dump(predictions_list, f)


# Function to plot images and their labels
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.0f}% (True: {})".format(
            predicted_label, 100 * np.max(predictions_array), true_label
        ),
        color=color,
    )


# Plot the first X test images, their predicted labels, and the true labels
# Change the value of num_rows and num_cols to display more or fewer images
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], y_test, x_test)
plt.tight_layout()
# plt.show()
plt.savefig("predictions.png")
