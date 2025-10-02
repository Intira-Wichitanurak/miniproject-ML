import joblib

model_path = r"C:\Users\ACER\PycharmProjects\PythonProject\FastAPI\app\xgb_unemployment_model.pkl"
model = joblib.load(model_path)

try:
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_
        print("Features in the model:")
        for f in features:
            print(f)
    else:
        print("This model does not store feature names. You may need to check training code.")
except Exception as e:
    print("Error checking features:", e)
