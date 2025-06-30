import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

# 1. Dados
(X_full, y_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_full = X_full / 255.0
X_test = X_test / 255.0

# 2. Separação aleatória com estratificação
X_train, X_valid, y_train, y_valid = train_test_split(
    X_full, y_full, test_size=5000, random_state=42, stratify=y_full
)

# 3. Adiciona canal para CNN
X_train = X_train[..., tf.newaxis]
X_valid = X_valid[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

# 4. Aumento de dados (mais agressivo agora)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.15
)
datagen.fit(X_train)

# 5. Modelo mais robusto
model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

# 6. Compila
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 7. Treinamento com early stopping
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=40,
    validation_data=(X_valid, y_valid),
    callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
)

# 8. Avaliação
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Acurácia final no teste: {test_acc * 100:.2f}%')
