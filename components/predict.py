def predict(model, input_features):
    prediction = model.predict([input_features])
    return prediction[0]