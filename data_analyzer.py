# data_analyzer.py
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple
import openai

class DataAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.data = None
        self.analysis_results = {}
        
    def analyze_dataset(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze new dataset structure and content using OpenAI API
        """
        self.data = data
        
        # Get basic dataset statistics
        basic_stats = self._get_basic_statistics()
        
        # Get column information
        columns_info = self._analyze_columns()
        
        # Use OpenAI to analyze dataset and suggest segmentation strategy
        ai_analysis = self._get_ai_data_analysis(basic_stats, columns_info)
        
        self.analysis_results = {
            "basic_stats": basic_stats,
            "columns_info": columns_info,
            "ai_analysis": ai_analysis
        }
        
        return self.analysis_results
    
    def _get_basic_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the dataset
        """
        return {
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "columns_list": self.data.columns.tolist(),
            "numeric_columns": self.data.select_dtypes(include=np.number).columns.tolist(),
            "categorical_columns": self.data.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_columns": self.data.select_dtypes(include=['datetime64']).columns.tolist(),
            "missing_values": self.data.isnull().sum().to_dict()
        }
    
    def _analyze_columns(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze each column in the dataset in detail
        """
        columns_info = {}
        
        for col in self.data.columns:
            col_info = {
                "data_type": str(self.data[col].dtype),
                "unique_values": self.data[col].nunique(),
                "missing_values": self.data[col].isnull().sum(),
                "missing_percentage": (self.data[col].isnull().sum() / len(self.data)) * 100
            }
            
            # For numeric columns
            if np.issubdtype(self.data[col].dtype, np.number):
                col_info.update({
                    "min": self.data[col].min(),
                    "max": self.data[col].max(),
                    "mean": self.data[col].mean(),
                    "median": self.data[col].median(),
                    "std": self.data[col].std()
                })
                
                # Check if likely to be categorical (few unique values)
                if self.data[col].nunique() < 10:
                    col_info["value_counts"] = self.data[col].value_counts().to_dict()
                    col_info["likely_categorical"] = True
                else:
                    col_info["likely_categorical"] = False
            
            # For categorical/text columns
            elif self.data[col].dtype == 'object' or self.data[col].dtype.name == 'category':
                if self.data[col].nunique() <= 50:  # Not too many unique values
                    col_info["value_counts"] = self.data[col].value_counts().head(20).to_dict()
                else:
                    col_info["sample_values"] = self.data[col].dropna().sample(min(5, len(self.data[col].dropna()))).tolist()
            
            columns_info[col] = col_info
            
        return columns_info
    
    def _get_ai_data_analysis(self, basic_stats: Dict[str, Any], columns_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use OpenAI to analyze dataset and suggest segmentation strategy
        """
        # Prepare data summary for OpenAI
        data_summary = {
            "basic_stats": basic_stats,
            "columns_sample": {k: v for k, v in columns_info.items() if k in list(columns_info.keys())[:10]}  # Limit to first 10 columns
        }
        
        # Create prompt for OpenAI
        prompt = f"""
        As a data scientist specializing in customer segmentation, analyze this dataset summary and provide recommendations:
        
        Dataset Overview:
        {json.dumps(data_summary, indent=2, default=str)}
        
        Please provide the following in JSON format:
        1. "dataset_type": What type of data does this appear to be (e.g. customer transactions, user behavior, demographics)?
        2. "segmentation_features": A list of column names that would be most useful for customer segmentation, with explanation for each
        3. "recommended_segmentation_method": Recommend best segmentation approach (K-Means, Hierarchical Clustering, etc.) with explanation
        4. "preprocessing_steps": Recommended preprocessing steps for the data
        5. "suggested_segments": Initial hypothesis about what customer segments might emerge
        6. "optimal_segment_count": Recommended number of segments to try first
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert data scientist specializing in customer analytics and segmentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        try:
            # Parse the response to get JSON content
            content = response.choices[0].message.content
            # Extract JSON part (in case there's text before or after)
            json_content = content
            if "```json" in content:
                json_content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_content = content.split("```")[1].split("```")[0]
                
            ai_analysis = json.loads(json_content.strip())
            return ai_analysis
        except Exception as e:
            # If parsing fails, return the raw response
            return {"error": str(e), "raw_response": response.choices[0].message.content}
    
    def get_recommended_segmentation_features(self) -> List[str]:
        """
        Get the columns recommended for segmentation
        """
        if not self.analysis_results or "ai_analysis" not in self.analysis_results:
            return []
        
        # Extract recommended features from AI analysis
        try:
            segmentation_features = []
            features_info = self.analysis_results["ai_analysis"].get("segmentation_features", [])
            
            if isinstance(features_info, list):
                # If it's a simple list of column names
                segmentation_features = features_info
            elif isinstance(features_info, dict):
                # If it's a dict with column names as keys
                segmentation_features = list(features_info.keys())
            
            # Make sure all features exist in the dataset
            existing_features = [f for f in segmentation_features if f in self.data.columns]
            return existing_features
        except:
            # Fallback to numeric columns if AI recommendation fails
            return self.analysis_results["basic_stats"]["numeric_columns"]
