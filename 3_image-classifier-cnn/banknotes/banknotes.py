import csv
import tensorflow as tf
import pprint as pp

from sklearn.model_selection import train_test_split

# Read data in from file
with open('banknotes.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    data = []

    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 1 if row[4] == "0" else 0 # Assuming '0' is real and '1' is counterfeit})
        })

# Separate data into training and testing groups
evidence = [row["evidence"]for row in data]
labels = [row["label"] for row in data]
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)

# Create a neural network
model = tf.keras.Sequential()

# Add a hidden layer with 8 units, with ReLU activation
model.add(tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)))

# Add output layer with 1 unit, with sigmoid activation
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Train neural network
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(
    x=tf.convert_to_tensor(X_training, dtype=tf.float32),
    y=tf.convert_to_tensor(y_training, dtype=tf.float32),
    epochs=20)


# Evaluate how well model performs
model.evaluate(
    x=tf.convert_to_tensor(X_testing, dtype=tf.float32),
    y=tf.convert_to_tensor(y_testing, dtype=tf.float32),
    verbose=2
)
input("Press Enter to exit...")



#X_training = tf.convert_to_tensor(X_training, dtype=tf.float32)
#y_training = tf.convert_to_tensor(y_training, dtype=tf.float32)
#X_testing = tf.convert_to_tensor(X_testing, dtype=tf.float32)
#y_testing = tf.convert_to_tensor(y_testing, dtype=tf.float32)
