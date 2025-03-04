import torch
import pandas as pd
from model import create_cnn_regression_model

# Load the model
input_shape = (1, 10, 1)  # Adjust this based on your actual input shape
model, criterion, optimizer = create_cnn_regression_model(input_shape)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Load the test dataset
test_data = pd.read_csv('test.csv')

# Preprocess the test data
test_ids = test_data['Id']
test_features = test_data.drop(columns=['Id']).values
test_features = test_features.reshape(-1, 1, 10, 1)  # Adjust this based on your actual input shape
test_features = torch.tensor(test_features, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    predictions = model(test_features).numpy()

# Save the predictions to a CSV file
output = pd.DataFrame({'Id': test_ids, 'Prediction': predictions.flatten()})
output.to_csv('predictions.csv', index=False)