from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = load_model('model_lstm_suhu.h5')

scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_input = float(request.form['temperature'])
    temp_scaled = scaler.fit_transform(np.array([[temp_input]]))
    temp_scaled = np.reshape(temp_scaled, (1, 1, 1))
    pred = model.predict(temp_scaled)
    pred_actual = scaler.inverse_transform(pred)
    return render_template('index.html',
                           prediction_text=f"🌤️ Prediksi Suhu Berikutnya di Melbourne: {pred_actual[0][0]:.2f}°C")

if __name__ == '__main__':
    app.run(debug=True)
