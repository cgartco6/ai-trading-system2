import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralNetworkModel(nn.Module):
    """Neural network model for trading predictions"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32], output_size: int = 3):
        super(NeuralNetworkModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return self.softmax(x)

class ModelManager:
    """Manages multiple ML models for trading predictions"""
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_performance: Dict[str, Dict] = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load existing models
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing models from disk"""
        try:
            for filename in os.listdir(self.model_dir):
                if filename.endswith('.pkl'):
                    model_name = filename[:-4]  # Remove .pkl extension
                    model_path = os.path.join(self.model_dir, filename)
                    
                    if 'scaler' in model_name:
                        self.scalers[model_name.replace('_scaler', '')] = joblib.load(model_path)
                    else:
                        self.models[model_name] = joblib.load(model_path)
                        
            logger.info(f"Loaded {len(self.models)} models and {len(self.scalers)} scalers")
        except Exception as e:
            logger.error(f"Error loading existing models: {e}")
    
    def create_ensemble(self, model_names: List[str], weights: Optional[List[float]] = None):
        """Create an ensemble of models"""
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        ensemble = {
            'models': [self.models[name] for name in model_names],
            'weights': weights,
            'model_names': model_names
        }
        
        self.models['ensemble'] = ensemble
        return ensemble
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, model_types: List[str] = None):
        """Train multiple models on the given data"""
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'xgboost', 'svm']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['default'] = scaler
        
        results = {}
        
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type} model...")
                
                if model_type == 'random_forest':
                    model = self._train_random_forest(X_train_scaled, y_train)
                elif model_type == 'gradient_boosting':
                    model = self._train_gradient_boosting(X_train_scaled, y_train)
                elif model_type == 'xgboost':
                    model = self._train_xgboost(X_train_scaled, y_train)
                elif model_type == 'svm':
                    model = self._train_svm(X_train_scaled, y_train)
                elif model_type == 'neural_network':
                    model = self._train_neural_network(X_train_scaled, y_train)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store model and performance
                self.models[model_type] = model
                self.model_performance[model_type] = {
                    'accuracy': accuracy,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'last_trained': datetime.now()
                }
                
                # Save model
                self._save_model(model_type, model)
                self._save_scaler('default', scaler)
                
                results[model_type] = accuracy
                logger.info(f"{model_type} model trained with accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                results[model_type] = None
        
        return results
    
    def _train_random_forest(self, X, y):
        """Train Random Forest model"""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_
    
    def _train_gradient_boosting(self, X, y):
        """Train Gradient Boosting model"""
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        
        model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_
    
    def _train_xgboost(self, X, y):
        """Train XGBoost model"""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        model = xgb.XGBClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_
    
    def _train_svm(self, X, y):
        """Train SVM model"""
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
        
        model = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_
    
    def _train_neural_network(self, X, y):
        """Train Neural Network model"""
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y.values)
        
        # Initialize model
        input_size = X.shape[1]
        model = NeuralNetworkModel(input_size=input_size)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        return model
    
    def predict(self, X: pd.DataFrame, model_name: str = 'ensemble') -> np.ndarray:
        """Make predictions using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Scale features
        if 'default' in self.scalers:
            X_scaled = self.scalers['default'].transform(X)
        else:
            X_scaled = X.values
        
        if model_name == 'ensemble':
            return self._predict_ensemble(X_scaled)
        else:
            model = self.models[model_name]
            
            if isinstance(model, NeuralNetworkModel):
                X_tensor = torch.FloatTensor(X_scaled)
                model.eval()
                with torch.no_grad():
                    predictions = model(X_tensor)
                    return torch.argmax(predictions, dim=1).numpy()
            else:
                return model.predict(X_scaled)
    
    def _predict_ensemble(self, X_scaled: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble"""
        ensemble = self.models['ensemble']
        predictions = []
        
        for model, weight in zip(ensemble['models'], ensemble['weights']):
            if isinstance(model, NeuralNetworkModel):
                X_tensor = torch.FloatTensor(X_scaled)
                model.eval()
                with torch.no_grad():
                    pred = model(X_tensor)
                    pred_np = torch.argmax(pred, dim=1).numpy()
            else:
                pred_np = model.predict(X_scaled)
            
            predictions.append(pred_np * weight)
        
        # Weighted voting
        ensemble_pred = np.round(np.sum(predictions, axis=0) / np.sum(ensemble['weights']))
        return ensemble_pred.astype(int)
    
    def predict_proba(self, X: pd.DataFrame, model_name: str = 'ensemble') -> np.ndarray:
        """Get prediction probabilities"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Scale features
        if 'default' in self.scalers:
            X_scaled = self.scalers['default'].transform(X)
        else:
            X_scaled = X.values
        
        model = self.models[model_name]
        
        if isinstance(model, NeuralNetworkModel):
            X_tensor = torch.FloatTensor(X_scaled)
            model.eval()
            with torch.no_grad():
                probabilities = model(X_tensor)
                return probabilities.numpy()
        else:
            return model.predict_proba(X_scaled)
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a model"""
        return self.model_performance.get(model_name, {})
    
    def _save_model(self, model_name: str, model: Any):
        """Save model to disk"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            
            if isinstance(model, NeuralNetworkModel):
                # Save PyTorch model state dict
                torch.save(model.state_dict(), model_path.replace('.pkl', '.pth'))
            else:
                joblib.dump(model, model_path)
                
            logger.info(f"Saved model: {model_name}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    def _save_scaler(self, scaler_name: str, scaler: StandardScaler):
        """Save scaler to disk"""
        try:
            scaler_path = os.path.join(self.model_dir, f"{scaler_name}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
        except Exception as e:
            logger.error(f"Error saving scaler {scaler_name}: {e}")
    
    def retrain_models(self, symbol: str = 'AAPL') -> bool:
        """Retrain models with new data"""
        try:
            # Fetch new data
            import yfinance as yf
            from signals.signal_generator import AISignalGenerator
            
            stock = yf.Ticker(symbol)
            data = stock.history(period='2y', interval='1d')
            
            if len(data) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                return False
            
            # Generate features and labels
            signal_generator = AISignalGenerator({})
            features = signal_generator._create_features(data)
            
            # Create target (1 if next day return positive, else 0)
            features['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            features = features.dropna()
            
            X = features.drop('target', axis=1)
            y = features['target']
            
            # Retrain models
            results = self.train_models(X, y)
            
            logger.info(f"Model retraining completed for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return False
