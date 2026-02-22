import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Path where your collected data is stored
DATA_PATH = "dataset"

# Load dataset
X = []
y = []

for label in os.listdir(DATA_PATH):
    label_path = os.path.join(DATA_PATH, label)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            data = np.load(os.path.join(label_path, file))
            X.append(data)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Build model
model = Sequential()
model.add(Dense(256, activation="relu", input_shape=(X.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(y_categorical.shape[1], activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("gesture_model.h5")

print("Model training completed and saved successfully!")
