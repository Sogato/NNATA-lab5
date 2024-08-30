import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from graphs import plot_training_history, plot_test_metrics

def process_and_split_data(file_path, window_size = 7):
    """
    Загрузка данных, их предварительная обработка, создание временных последовательностей и разделение на обучающую,
    валидационную и тестовую выборки.

    Аргументы:
        file_path (str): Путь к CSV файлу с данными.
        window_size (int): Размер окна для создания временных последовательностей.

    Возвращает:
        X_train, X_val, X_test (numpy.ndarray): Массивы с данными для обучения, валидации и тестирования.
        y_train, y_val, y_test (numpy.ndarray): Массивы с метками для обучения, валидации и тестирования.
    """
    # Загрузка данных из CSV файла с указанием кодировки (для кириллицы)
    df = pd.read_csv(file_path, delimiter=';', encoding='cp1251')

    df['LocalTime'] = pd.to_datetime(df['LocalTime'], format='%d.%m.%Y %H:%M')
    df = df.sort_values('LocalTime')

    # Удаление ненужных колонок
    df = df.drop(columns=['P0', 'P', 'U', 'DD'])
    df = df.dropna()

    # Нормализация данных в колонке 'T'
    df['T'] = (df['T'] - df['T'].min()) / (df['T'].max() - df['T'].min())

    # Создание временных окон для последовательностей
    sequences = []
    labels = []
    for i in range(len(df) - window_size):
        seq = df['T'].iloc[i:i + window_size].values  # Временное окно
        label = df['T'].iloc[i + window_size]  # Следующее значение (метка)
        sequences.append(seq)
        labels.append(label)

    sequences = np.array(sequences)
    labels = np.array(labels)

    # Разделение данных на обучающую, валидационную и тестовую выборки
    X_train, X_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_mlp_model(input_shape, neurons, dropout):
    """Создание модели MLP с параметрами neurons и dropout."""
    model = Sequential([
        Input(shape=input_shape),
        Dense(neurons, activation='relu'),
        Dropout(dropout),
        Dense(neurons, activation='relu'),
        Dense(1)  # Регрессия, поэтому один выходной нейрон
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_rnn_model(input_shape, neurons, dropout):
    """Создание модели RNN с параметрами neurons и dropout."""
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(neurons, activation='relu'),
        Dropout(dropout),
        Dense(neurons, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_lstm_model(input_shape, neurons, dropout):
    """Создание модели LSTM с параметрами neurons и dropout."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(neurons, activation='relu'),
        Dropout(dropout),
        Dense(neurons, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def hyperparameter_search(create_model_func, input_shape, X_train, y_train, X_val, y_val, X_test, y_test, model_type):
    """
    Подбор гиперпараметров для любой архитектуры модели.

    Аргументы:
        create_model_func (function): Функция для создания модели (MLP, RNN, LSTM).
        input_shape (tuple): Форма входных данных.
        X_train, y_train (numpy.ndarray): Обучающие данные и метки.
        X_val, y_val (numpy.ndarray): Валидационные данные и метки.
        X_test, y_test (numpy.ndarray): Тестовые данные и метки.
        model_type (str): Тип модели (MLP, RNN, LSTM).

    Возвращает:
        best_model (tf.keras.Model): Лучшая модель с подобранными гиперпараметрами.
    """
    # Список гиперпараметров для перебора
    neurons_options = [32, 64, 128, 256, 512, 1024]
    dropout_options = [0.1, 0.2, 0.3, 0.4, 0.5]

    best_mse = float("inf")
    best_model = None
    best_history = None
    best_params = {}

    print(f"Начало подбора гиперпараметров для {model_type} модели...")

    for neurons in neurons_options:
        for dropout in dropout_options:
            print(f"Пробуем конфигурацию: {neurons} нейронов, Dropout {dropout}")

            # Создание модели с текущими гиперпараметрами
            model = create_model_func(input_shape, neurons, dropout)

            # Обучаем модель
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=0)

            # Оценка модели на валидационных данных
            mse, mae, r2 = evaluate_model(model, X_val, y_val)
            print(f"{model_type} Валидация - MSE: {mse}, MAE: {mae}, R²: {r2}")

            # Сохраняем лучшую модель
            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_history = history
                best_params = {'neurons': neurons, 'dropout': dropout}

    print(f"\nПодбор гиперпараметров для {model_type} модели завершен.")
    print(f"Лучшая конфигурация: {best_params['neurons']} нейронов, Dropout {best_params['dropout']}")
    mse, mae, r2 = evaluate_model(best_model, X_test, y_test)
    metrics = {'MSE': mse, 'MAE': mae, 'R²': r2}
    print(f"{model_type} Тестирование - MSE: {mse}, MAE: {mae}, R²: {r2}")

    plot_training_history(best_history, model_type)
    plot_test_metrics(metrics, model_type)

    print("\nВсе исследованные конфигурации:")
    for neurons in neurons_options:
        for dropout in dropout_options:
            print(f"- {neurons} нейронов, Dropout {dropout}")

    print(f"\nПараметры лучшей модели: {best_params['neurons']} нейронов, Dropout {best_params['dropout']}")
    print(f"Лучшая модель - MSE: {mse}, MAE: {mae}, R²: {r2}")

    return best_model


def evaluate_model(model, X_test, y_test):
    """
    Оценка модели на тестовых данных.

    Аргументы:
        model (tf.keras.Model): Обученная модель.
        X_test, y_test (numpy.ndarray): Тестовые данные и метки.

    Возвращает:
        mse (float): Среднеквадратичная ошибка.
        mae (float): Средняя абсолютная ошибка.
        r2 (float): Коэффициент детерминации R².
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, mae, r2


if __name__ == "__main__":
    window_size = 7  # Размер окна для временных последовательностей

    # Путь к файлу с данными
    file_path = 'data/Volgograd_weather_15102020_15102012.csv'

    # Обработка данных и их разделение на выборки
    X_train, X_val, X_test, y_train, y_val, y_test = process_and_split_data(file_path)

    # Вывод размеров выборок
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер валидационной выборки: {X_val.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    # Добавляем дополнительное измерение для RNN и LSTM
    X_train_rnn = np.expand_dims(X_train, -1)
    X_val_rnn = np.expand_dims(X_val, -1)
    X_test_rnn = np.expand_dims(X_test, -1)

    # Подбор гиперпараметров для MLP
    print("\n\nПодбор гиперпараметров для MLP модели...")
    best_mlp_model = hyperparameter_search(create_mlp_model, (window_size,), X_train, y_train, X_val, y_val, X_test,
                                           y_test, "MLP")

    # Подбор гиперпараметров для RNN
    print("\n\nПодбор гиперпараметров для RNN модели...")
    best_rnn_model = hyperparameter_search(create_rnn_model, (window_size, 1), X_train_rnn, y_train, X_val_rnn, y_val,
                                           X_test_rnn, y_test, "RNN")

    # Подбор гиперпараметров для LSTM
    print("\n\nПодбор гиперпараметров для LSTM модели...")
    best_lstm_model = hyperparameter_search(create_lstm_model, (window_size, 1), X_train_rnn, y_train, X_val_rnn, y_val,
                                            X_test_rnn, y_test, "LSTM")
