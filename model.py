import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Загрузка датасета
df = pd.read_csv("personality_dataset.csv")

# Перевод Yes/No значений в числовые (1/0), потому что нейронные сети требуют числовые данные
df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Удаление строк с пропущенными значениями, поскольку они могут вызвать ошибки при обучении модели
df = df.dropna()

# Значения, которые мы хотим использовать в качестве признаков
features = df.drop('Personality', axis=1)
# Значение, которое мы хотим предсказать
labels = df['Personality']

# Трансформация лейблов в числовые значения (Introvert = 0, Extrovert = 1)
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
# Сохранение энкодера для последующего использования в flask приложении
joblib.dump(encoder, 'encoder.joblib')

# Трансформация признаков в Numpy массив, чтобы их можно было использовать в Keras (0-1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# Сохранение скейлера для последующего использования в flask приложении
joblib.dump(scaler, 'scaler.joblib')

# Разделение данных на обучающую и валидационную выборки. 20% данных будет использоваться для валидации, 80% для обучения. 
# random_state=42 обеспечивает воспроизводимость результатов. Каждый раз при запуске кода данные будут разделены одинаково.
features_train, features_test, labels_train, labels_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)

# Создание и обучение модели нейронной сети
# Используем Sequential модель с тремя слоями: два скрытых слоя с 16 и 8 нейронами и выходной слой с 2 нейронами (для двух классов: Introvert и Extrovert).
# Мы используем ReLU активацию для скрытых слоев и softmax для выходного слоя, чтобы получить вероятности классов.
# X_train.shape[1] - это количество признаков, которые мы используем для обучения модели (количество колонок).
model = Sequential([
    Dense(16, activation='relu', input_shape=(features_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax')
])

# Вывод структуры модели
# Типы слоев, количество нейронов в каждом слое, количество параметров, которые будут обучаться.
model.summary()

# Компиляция модели с использованием Adam оптимизатора и кросс-энтропии как функции потерь (используется когда лейблы целые числа). accuracy - метрика для оценки качества модели.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Обучение модели на обучающей выборке с валидацией на 20% данных.
# epochs=30 - модель будет обучаться 30 эпох, batch_size=10 - количество примеров в одной итерации обучения. verbose=0 - отключает вывод информации об обучении.
history = model.fit(features_train, labels_train, epochs=30, batch_size=10, validation_split=0.2, verbose=0)

# Сохранение модели в файл 'model.h5' для последующего использования в flask приложении.
model.save('model.h5')

# ========== МЕТРИКИ и ВИЗУАЛИЗАЦИЯ ==========

# Предсказания
probs = model.predict(features_test)
predicted = np.argmax(probs, axis=1)

# Матрица ошибок
cm = confusion_matrix(labels_test, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
