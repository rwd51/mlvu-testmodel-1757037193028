# model.py - Sample ML Model for Testing MLVU Deployment
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib
import os

class MLModel:
    def __init__(self):
        self.model = None
        self.model_type = "classification"
        
    def load_model(self):
        """Load the trained model - creates one if doesn't exist"""
        model_file = 'iris_model.pkl'
        
        if os.path.exists(model_file):
            # Load existing model
            self.model = joblib.load(model_file)
            print(f"Loaded existing model from {model_file}")
        else:
            # Create and train a simple model for testing
            print("No model file found, creating a simple Iris classifier...")
            iris = load_iris()
            X, y = iris.data, iris.target
            
            # Train a simple RandomForest
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            self.model.fit(X, y)
            
            # Save the model
            joblib.dump(self.model, model_file)
            print(f"Created and saved model to {model_file}")
            
        return self.model
    
    def predict(self, input_data):
        """Make predictions on input data"""
        if self.model is None:
            self.load_model()
        
        # Convert input to numpy array if needed
        if isinstance(input_data, pd.DataFrame):
            X = input_data.values
        else:
            X = np.array(input_data)
            
        predictions = self.model.predict(X)
        return predictions
    
    def get_input_schema(self):
        """Define expected input format for the Iris dataset"""
        return {
            "sepal_length": {
                "type": "number", 
                "min": 4.0, 
                "max": 8.0, 
                "default": 5.8,
                "description": "Sepal length in cm"
            },
            "sepal_width": {
                "type": "number", 
                "min": 2.0, 
                "max": 4.5, 
                "default": 3.0,
                "description": "Sepal width in cm"
            },
            "petal_length": {
                "type": "number", 
                "min": 1.0, 
                "max": 7.0, 
                "default": 3.8,
                "description": "Petal length in cm"
            },
            "petal_width": {
                "type": "number", 
                "min": 0.1, 
                "max": 2.5, 
                "default": 1.2,
                "description": "Petal width in cm"
            }
        }
    
    def get_output_schema(self):
        """Define output format"""
        return {
            "prediction": {
                "type": "number",
                "description": "Predicted iris species (0=setosa, 1=versicolor, 2=virginica)"
            },
            "species_name": {
                "type": "string",
                "description": "Human-readable species name"
            }
        }

# Optional: Test the model locally
if __name__ == "__main__":
    # Test the model
    model = MLModel()
    model.load_model()
    
    # Test prediction
    test_input = pd.DataFrame({
        'sepal_length': [5.8],
        'sepal_width': [3.0], 
        'petal_length': [3.8],
        'petal_width': [1.2]
    })
    
    prediction = model.predict(test_input)
    print(f"Test prediction: {prediction[0]}")
    
    species_names = ['setosa', 'versicolor', 'virginica']
    print(f"Predicted species: {species_names[prediction[0]]}")