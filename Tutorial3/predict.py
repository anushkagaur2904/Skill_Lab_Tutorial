"""
Use the trained model on new students.

  python predict.py study_hours=6 attendance=85
  python predict.py study_hours=2 attendance=60 prev_grade=55   # if trained with it
"""
import json
import sys

import joblib
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("student_performance_model.keras")
scaler = joblib.load("scaler.joblib")
cfg = json.load(open("model_config.json"))

given = dict(arg.split("=") for arg in sys.argv[1:])
missing = [f for f in cfg["features"] if f not in given]
if missing:
    sys.exit(f"Missing values for: {missing}\nExpected features: {cfg['features']}")

x = scaler.transform(np.array([[float(given[f]) for f in cfg["features"]]]))
prob = float(model.predict(x, verbose=0)[0, 0])
print(f"{'PASS' if prob > cfg['threshold'] else 'FAIL'} (P(pass) = {prob:.2f})")
