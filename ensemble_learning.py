import numpy as np

def ensemble_predict(models, test_data):
    # Extracting probabilistic values from each model
    model_outputs = [model.predict(test_data) for model in models]

    # Assuming two best-performing models: Xception and DenseNet-169
    xception_output = model_outputs[0]  # Probabilistic output from Xception model
    densenet_output = model_outputs[1]  # Probabilistic output from DenseNet-169 model

    # Performing majority voting
    combined_predictions = np.argmax(xception_output + densenet_output, axis=1)
    
    return combined_predictions
