# segmentation_model.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import json
from typing import Dict, List, Any, Tuple

class DynamicSegmentationModel:
    def __init__(self):
        self.scaler = None
        self.encoder = None
        self.model = None
        self.pca = None
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        self.segment_features = None
        self.segment_profiles = None
        self.segment_counts = None
        
    def fit(self, data: pd.DataFrame, ai_recommendations: Dict[str, Any]) -> 'DynamicSegmentationModel':
        """
        Fit segmentation model based on AI recommendations
        """
        # Get features and algorithm from AI recommendations
        self.segment_features = ai_recommendations.get("segmentation_features", [])
        algorithm = ai_recommendations.get("recommended_segmentation_method", "kmeans").lower()
        optimal_segments = ai_recommendations.get("optimal_segment_count", 4)
        
        # If no features recommended or features don't exist in data, use all numeric columns
        if not self.segment_features or not all(feat in data.columns for feat in self.segment_features):
            self.segment_features = data.select_dtypes(include=np.number).columns.tolist()
        
        # Identify categorical and numerical features
        self.categorical_features = []
        self.numerical_features = []
        
        for feat in self.segment_features:
            if feat in data.columns:
                if data[feat].dtype == 'object' or data[feat].dtype.name == 'category' or (
                    data[feat].dtype in [np.int64, np.int32] and data[feat].nunique() < 20
                ):
                    self.categorical_features.append(feat)
                elif np.issubdtype(data[feat].dtype, np.number):
                    self.numerical_features.append(feat)
        
        # Clone the input data to avoid modifying it
        processed_data = data.copy()
        
        # Handle missing values
        for feature in self.numerical_features:
            processed_data[feature].fillna(processed_data[feature].median(), inplace=True)
            
        for feature in self.categorical_features:
            processed_data[feature].fillna(processed_data[feature].mode()[0], inplace=True)
        
        # Create feature matrix X
        X = self._preprocess_data(processed_data)
        
        # Determine number of clusters
        n_clusters = self._determine_optimal_clusters(X, optimal_segments)
        
        # Select and fit clustering model based on AI recommendation
        if "hierarchical" in algorithm:
            self.model = AgglomerativeClustering(n_clusters=n_clusters)
        elif "dbscan" in algorithm:
            self.model = DBSCAN(eps=0.5, min_samples=5)
        elif "gaussian" in algorithm or "gmm" in algorithm:
            self.model = GaussianMixture(n_components=n_clusters, random_state=42)
        else:
            # Default to KMeans
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # Fit the model
        if hasattr(self.model, 'fit_predict'):
            clusters = self.model.fit_predict(X)
        else:
            self.model.fit(X)
            clusters = self.model.predict(X)
        
        # Add cluster labels to data
        processed_data['segment'] = clusters
        
        # Create segment profiles
        self.segment_profiles = self._create_segment_profiles(processed_data)
        self.segment_counts = processed_data['segment'].value_counts().to_dict()
        
        # Fit PCA for visualization
        self.pca = PCA(n_components=2)
        self.pca.fit(X)
        
        return self
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data for clustering
        """
        # Handle numerical features
        if self.numerical_features:
            self.scaler = StandardScaler()
            numerical_data = self.scaler.fit_transform(data[self.numerical_features])
            
            # Convert to DataFrame for easier concatenation
            numerical_df = pd.DataFrame(
                numerical_data, 
                columns=self.numerical_features,
                index=data.index
            )
        else:
            numerical_df = pd.DataFrame(index=data.index)
        
        # Handle categorical features
        if self.categorical_features:
            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            categorical_data = self.encoder.fit_transform(data[self.categorical_features])
            
            # Get feature names for the encoded columns
            if hasattr(self.encoder, 'get_feature_names_out'):
                encoded_feature_names = self.encoder.get_feature_names_out(self.categorical_features)
            else:
                encoded_feature_names = [f"{feat}_{cat}" for feat in self.categorical_features 
                                        for cat in self.encoder.categories_[self.categorical_features.index(feat)]]
            
            # Convert to DataFrame for easier concatenation
            categorical_df = pd.DataFrame(
                categorical_data, 
                columns=encoded_feature_names,
                index=data.index
            )
        else:
            categorical_df = pd.DataFrame(index=data.index)
        
        # Combine numerical and categorical features
        combined_df = pd.concat([numerical_df, categorical_df], axis=1)
        
        # Save feature names for future reference
        self.feature_names = combined_df.columns.tolist()
        
        return combined_df.values
    
    def _determine_optimal_clusters(self, X: np.ndarray, initial_guess: int) -> int:
        """
        Determine the optimal number of clusters using silhouette score
        """
        if len(X) < 1000:  # Only try multiple clusters for smaller datasets
            max_clusters = min(10, len(X) // 10)  # Don't try more clusters than 1/10th of the data points
            
            if max_clusters <= 2:
                return max(2, initial_guess)  # If dataset is small, use initial guess or at least 2
                
            # Try different cluster counts
            silhouette_scores = []
            cluster_range = range(2, max_clusters + 1)
            
            for n_clusters in cluster_range:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                
                try:
                    score = silhouette_score(X, labels)
                    silhouette_scores.append((n_clusters, score))
                except:
                    continue
            
            if silhouette_scores:
                # Get the cluster count with the highest score
                best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
                return best_n_clusters
        
        # If we can't determine optimal clusters, use the initial guess
        return initial_guess
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict segments for new data
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        
        # Clone the input data to avoid modifying it
        processed_data = data.copy()
        
        # Handle missing values
        for feature in self.numerical_features:
            if feature in processed_data.columns:
                processed_data[feature].fillna(processed_data[feature].median(), inplace=True)
            
        for feature in self.categorical_features:
            if feature in processed_data.columns:
                processed_data[feature].fillna(processed_data[feature].mode()[0], inplace=True)
        
        # Create feature matrix X
        X = self._preprocess_data(processed_data)
        
        # Predict clusters
        if hasattr(self.model, 'predict'):
            clusters = self.model.predict(X)
        else:
            clusters = self.model.fit_predict(X)
        
        # Add cluster labels to data
        processed_data['segment'] = clusters
        
        return processed_data
    
    def _create_segment_profiles(self, data_with_segments: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Create profiles for each segment based on the features used for segmentation
        """
        segment_profiles = {}
        
        for segment in sorted(data_with_segments['segment'].unique()):
            segment_data = data_with_segments[data_with_segments['segment'] == segment]
            profile = {
                "segment_id": int(segment),
                "size": len(segment_data),
                "percentage": (len(segment_data) / len(data_with_segments)) * 100,
                "numerical_features": {},
                "categorical_features": {}
            }
            
            # Add numerical feature statistics
            for feature in self.numerical_features:
                if feature in segment_data.columns:
                    profile["numerical_features"][feature] = {
                        "mean": segment_data[feature].mean(),
                        "median": segment_data[feature].median(),
                        "min": segment_data[feature].min(),
                        "max": segment_data[feature].max(),
                        "std": segment_data[feature].std()
                    }
            
            # Add categorical feature distributions
            for feature in self.categorical_features:
                if feature in segment_data.columns:
                    value_counts = segment_data[feature].value_counts()
                    total = value_counts.sum()
                    
                    profile["categorical_features"][feature] = {
                        "top_values": value_counts.head(5).to_dict(),
                        "top_percentages": (value_counts.head(5) / total * 100).to_dict()
                    }
            
            segment_profiles[int(segment)] = profile
        
        return segment_profiles
    
    def get_pca_projection(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get PCA projection for visualization
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet")
        
        # Preprocess data
        processed_data = data.copy()
        
        # Handle missing values
        for feature in self.numerical_features:
            if feature in processed_data.columns:
                processed_data[feature].fillna(processed_data[feature].median(), inplace=True)
            
        for feature in self.categorical_features:
            if feature in processed_data.columns:
                processed_data[feature].fillna(processed_data[feature].mode()[0], inplace=True)
        
        # Create feature matrix X
        X = self._preprocess_data(processed_data)
        
        # Get PCA projection
        return self.pca.transform(X)
