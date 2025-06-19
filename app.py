from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

# Инициализация Flask приложения
app = Flask(__name__)

# Загрузка модели и необходимых объектов. scaler - это StandardScaler, который мы использовали для нормализации данных, encoder - это LabelEncoder, который мы использовали для кодирования меток классов.
model = tf.keras.models.load_model('model.h5')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')

# Список полей, которые мы будем использовать в форме ввода. Эти поля соответствуют признакам, которые мы использовали для обучения модели. 
INPUT_FIELDS = [
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]

# Функция для преобразования значений 'Yes'/'No' в 1/0
def yes_no_to_int(value):
    return 1 if value.strip().lower() == 'yes' else 0

# Главная страница приложения, которая обрабатывает GET и POST запросы.
# При GET запросе отображается форма ввода, при POST запросе обрабатываются данные из формы, выполняется предсказание и отображается результат.
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    # Если запрос POST, то обрабатываем данные из формы
    if request.method == 'POST':
        try:
            features = []
            for field in INPUT_FIELDS:
                val = request.form[field]
                # Если поле - это 'Stage_fear' или 'Drained_after_socializing', то преобразуем значение в 1/0
                if field in ['Stage_fear', 'Drained_after_socializing']:
                    if val.lower() not in ['yes', 'no']:
                        raise ValueError(f"Invalid value for {field}, must be Yes or No")
                    features.append(yes_no_to_int(val))
                # Иначе преобразуем значение в float и проверяем его диапазон
                else:
                    val_float = float(val)
                    if field != 'Friends_circle_size' and not (0 <= val_float <= 10):
                        raise ValueError(f"{field.replace('_', ' ')} must be between 0 and 10")
                    if field == 'Friends_circle_size' and val_float < 0:
                        raise ValueError(f"Friends circle size must be zero or positive")
                    features.append(val_float)

            # Преобразуем список признаков в Numpy массив и нормализуем его с помощью scaler
            input_scaled = scaler.transform([features])
            # Выполняем предсказание с помощью модели
            probs = model.predict(input_scaled)[0]
            # Получаем индекс класса с максимальной вероятностью
            # и преобразуем его в метку с помощью encoder
            class_idx = np.argmax(probs)
            # Получаем метку класса и вероятность
            label = encoder.inverse_transform([class_idx])[0]
            # Формируем строку с предсказанием и вероятностью
            confidence = probs[class_idx] * 100
            prediction = f"{label} (Confidence: {confidence:.1f}%)"

        except Exception as e:
            error = str(e)

    # Если запрос GET, то просто отображаем форму ввода
    # если есть предсказание или ошибка, то передаем их в шаблон для отображения (шаблон index.html должен быть создан в папке templates)
    return render_template('index.html', prediction=prediction, error=error)

# Запуск Flask приложения
# Если этот файл запущен напрямую, то запускаем приложение в режиме отладки
if __name__ == '__main__':
    app.run(debug=True)
