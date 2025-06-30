import tensorflow as tf
from tensorflow import keras

# 1. Dados normalizados
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# 2. Separação aleatória de validação
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=5000, random_state=42, stratify=y_train_full
)

# 3. Adiciona o canal para CNN: (28, 28, 1)
X_train = X_train[..., tf.newaxis]
X_valid = X_valid[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

# 4. Arquitetura CNN
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.4),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

optimizer = keras.optimizers.Adam(learning_rate=0.0005)

# 5. Compila o modelo
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. Treinamento
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=64,
    validation_data=(X_valid, y_valid),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)

# 7. Avaliação
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Acurácia no teste: {test_acc * 100:.2f}%")
