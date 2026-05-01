import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import xgboost as xgb
from typing import Dict, List, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')


class RiskPredictor:
    
    def __init__(self, model_type: str = 'classification'):
        
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
    
    def prepare_features(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        volume: Optional[pd.DataFrame] = None,
        lookback_window: int = 30
    ) -> pd.DataFrame:
        
        features_list = []
        
        for ticker in returns.columns:
            ticker_features = pd.DataFrame(index=returns.index)
            
            ticker_features[f'{ticker}_rolling_vol_10'] = returns[ticker].rolling(10).std() * np.sqrt(252)
            ticker_features[f'{ticker}_rolling_vol_30'] = returns[ticker].rolling(30).std() * np.sqrt(252)
            ticker_features[f'{ticker}_rolling_vol_60'] = returns[ticker].rolling(60).std() * np.sqrt(252)
            
            ticker_features[f'{ticker}_ewma_vol'] = returns[ticker].ewm(span=30).std() * np.sqrt(252)
            
            ticker_features[f'{ticker}_ma_5'] = prices[ticker].rolling(5).mean()
            ticker_features[f'{ticker}_ma_20'] = prices[ticker].rolling(20).mean()
            ticker_features[f'{ticker}_ma_50'] = prices[ticker].rolling(50).mean()
            
            ticker_features[f'{ticker}_price_to_ma5'] = prices[ticker] / ticker_features[f'{ticker}_ma_5']
            ticker_features[f'{ticker}_price_to_ma20'] = prices[ticker] / ticker_features[f'{ticker}_ma_20']
            
            ticker_features[f'{ticker}_momentum_5'] = returns[ticker].rolling(5).sum()
            ticker_features[f'{ticker}_momentum_10'] = returns[ticker].rolling(10).sum()
            
            ticker_features[f'{ticker}_return_1d'] = returns[ticker]
            ticker_features[f'{ticker}_return_5d'] = returns[ticker].rolling(5).mean()
            ticker_features[f'{ticker}_return_20d'] = returns[ticker].rolling(20).mean()
            
            if volume is not None and ticker in volume.columns:
                ticker_features[f'{ticker}_volume_change'] = volume[ticker].pct_change()
                ticker_features[f'{ticker}_volume_ma_ratio'] = volume[ticker] / volume[ticker].rolling(20).mean()
            
            delta = prices[ticker].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            ticker_features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
            
            cumulative = (1 + returns[ticker]).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            ticker_features[f'{ticker}_drawdown'] = drawdown
            
            features_list.append(ticker_features)
        
        all_features = pd.concat(features_list, axis=1)
        
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.ffill().bfill()
        
        return all_features
    
    def create_target_classification(
        self,
        returns: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
        thresholds: Tuple[float, float] = (0.01, 0.02)
    ) -> pd.Series:
        
        if weights is None:
            weights = np.array([1.0 / len(returns.columns)] * len(returns.columns))
        
        portfolio_returns = (returns * weights).sum(axis=1)
        
        rolling_vol = portfolio_returns.rolling(30).std() * np.sqrt(252)
        
        target = pd.Series(index=returns.index, dtype=int)
        target[rolling_vol <= thresholds[0]] = 0
        target[(rolling_vol > thresholds[0]) & (rolling_vol <= thresholds[1])] = 1
        target[rolling_vol > thresholds[1]] = 2
        
        target = target.shift(-1)
        
        return target
    
    def create_target_regression(
        self,
        returns: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
        forecast_horizon: int = 5
    ) -> pd.Series:
        
        if weights is None:
            weights = np.array([1.0 / len(returns.columns)] * len(returns.columns))
        
        portfolio_returns = (returns * weights).sum(axis=1)
        
        future_volatility = portfolio_returns.rolling(forecast_horizon).std().shift(-forecast_horizon) * np.sqrt(252)
        
        return future_volatility
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        use_xgboost: bool = False,
        hyperparameters: Optional[Dict] = None
    ) -> Dict:
        
        data = pd.concat([X, y], axis=1).dropna()
        X_clean = data[X.columns]
        y_clean = data[y.name] if y.name else data.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=test_size, shuffle=False
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = X.columns.tolist()
        
        if self.model_type == 'classification':
            if use_xgboost:
                default_params = {
                    'objective': 'multi:softmax',
                    'num_class': 3,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'n_estimators': 100
                }
                params = {**default_params, **(hyperparameters or {})}
                self.model = xgb.XGBClassifier(**params)
            else:
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
                params = {**default_params, **(hyperparameters or {})}
                self.model = RandomForestClassifier(**params)
        else:
            if use_xgboost:
                default_params = {
                    'objective': 'reg:squarederror',
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'n_estimators': 100
                }
                params = {**default_params, **(hyperparameters or {})}
                self.model = xgb.XGBRegressor(**params)
            else:
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
                params = {**default_params, **(hyperparameters or {})}
                self.model = RandomForestRegressor(**params)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_test_scaled)
        
        if self.model_type == 'classification':
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            results = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'predictions': y_pred,
                'actual': y_test,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
        else:
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results = {
                'train_r2': train_score,
                'test_r2': test_score,
                'mse': mse,
                'rmse': rmse,
                'predictions': y_pred,
                'actual': y_test
            }
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = feature_importance
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        X_scaled = self.scaler.transform(X[self.feature_names])
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_risk_level(self, X: pd.DataFrame) -> List[str]:
        
        predictions = self.predict(X)
        
        risk_map = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
        risk_levels = [risk_map.get(pred, 'UNKNOWN') for pred in predictions]
        
        return risk_levels
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_importance
    
    def save_model(self, filepath: str):
        
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict:
        
        data = pd.concat([X, y], axis=1).dropna()
        X_clean = data[X.columns]
        y_clean = data[y.name] if y.name else data.iloc[:, -1]
        
        X_scaled = self.scaler.fit_transform(X_clean)
        
        if self.model is None:
            if self.model_type == 'classification':
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        scores = cross_val_score(self.model, X_scaled, y_clean, cv=cv)
        
        return {
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
