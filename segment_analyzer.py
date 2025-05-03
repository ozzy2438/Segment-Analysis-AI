# segment_analyzer.py
import openai
import json
from typing import Dict, List, Any
import pandas as pd

class SegmentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key
        
    def analyze_segments(self, 
                        segment_profiles: Dict[int, Dict], 
                        original_data: pd.DataFrame, 
                        ai_data_analysis: Dict[str, Any]) -> Dict[int, Dict]:
        """
        Use OpenAI API to generate insightful analysis and recommendations for each segment
        """
        segment_analysis = {}
        dataset_type = ai_data_analysis.get("dataset_type", "customer data")
        
        for segment_id, profile in segment_profiles.items():
            # Extract segment data
            segment_data = original_data[original_data['segment'] == segment_id]
            
            # Get segment analysis from OpenAI
            segment_analysis[segment_id] = self._get_segment_insights(
                segment_id, 
                profile, 
                segment_data,
                dataset_type
            )
        
        return segment_analysis
    
    def _get_segment_insights(self, 
                            segment_id: int, 
                            profile: Dict, 
                            segment_data: pd.DataFrame,
                            dataset_type: str) -> Dict:
        """
        Generate insights for a specific segment using OpenAI
        """
        # Prepare segment profile data for the prompt
        numerical_insights = []
        for feature, stats in profile.get("numerical_features", {}).items():
            numerical_insights.append(f"- {feature}: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}")
        
        categorical_insights = []
        for feature, stats in profile.get("categorical_features", {}).items():
            top_values = list(stats.get("top_values", {}).items())
            if top_values:
                top_value_str = f"- {feature} - Most common: {top_values[0][0]} ({top_values[0][1]} occurrences)"
                categorical_insights.append(top_value_str)
        
        # Create sample data rows for context
        sample_rows = segment_data.head(5).to_dict(orient='records')
        sample_rows_str = json.dumps(sample_rows, default=str)
        
        # Create prompt for OpenAI
        prompt = f"""
        As a marketing and customer analytics expert, analyze this customer segment from {dataset_type}:
        
        Segment ID: {segment_id}
        Size: {profile['size']} customers ({profile['percentage']:.2f}% of total)
        
        Numerical Feature Statistics:
        {chr(10).join(numerical_insights)}
        
        Categorical Feature Distributions:
        {chr(10).join(categorical_insights)}
        
        Sample data points from this segment:
        {sample_rows_str}
        
        Based on this information, please provide:
        
        1. A descriptive name for this segment
        2. Key characteristics that define this segment
        3. Customer behavioral insights (purchasing habits, preferences, etc.)
        4. 3-5 specific marketing strategy recommendations tailored to this segment
        5. The preferred communication channels for this segment
        6. Suggested products or services to promote to this segment
        
        Return your analysis in JSON format with these keys: 
        "segment_name", "key_characteristics", "behavioral_insights", "marketing_recommendations", "preferred_channels", "product_recommendations"
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert marketing analyst with deep knowledge of customer segmentation and targeted marketing strategies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1200
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
                
            segment_insights = json.loads(json_content.strip())
            return segment_insights
        except Exception as e:
            # If parsing fails, create a structured response from the raw text
            raw_response = response.choices[0].message.content
            
            # Try to extract sections based on keywords
            segment_name = "Segment " + str(segment_id)
            if "segment name" in raw_response.lower():
                segment_name_parts = raw_response.split("segment name", 1)[1].split("\n", 1)
                if segment_name_parts:
                    segment_name = segment_name_parts[0].strip(": ")
                    
            return {
                "segment_name": segment_name,
                "key_characteristics": raw_response,
                "behavioral_insights": "Analysis not available in structured format. Please see raw analysis.",
                "marketing_recommendations": "See full analysis for recommendations.",
                "preferred_channels": [],
                "product_recommendations": [],
                "raw_analysis": raw_response
            }
    
    def generate_campaign_ideas(self, segment_analysis: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Generate overall campaign strategy based on all segments
        """
        segments_summary = []
        
        for segment_id, analysis in segment_analysis.items():
            segments_summary.append({
                "segment_id": segment_id,
                "segment_name": analysis.get("segment_name", f"Segment {segment_id}"),
                "key_characteristics": analysis.get("key_characteristics", [])[:3],
                "preferred_channels": analysis.get("preferred_channels", [])
            })
        
        prompt = f"""
        As a marketing campaign strategist, review these customer segments:
        
        {json.dumps(segments_summary, indent=2)}
        
        Based on these segments, please provide:
        
        1. An overall marketing campaign strategy that addresses all segments
        2. Ideas for cross-segment campaigns that can appeal to multiple segments
        3. A timeline recommendation for rolling out these campaigns
        4. KPI suggestions for measuring campaign effectiveness
        5. Budget allocation recommendations across segments
        
        Return your strategy in JSON format with these keys: 
        "overall_strategy", "cross_segment_campaigns", "timeline_recommendation", "kpi_suggestions", "budget_allocation"
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert marketing strategist specializing in campaign development across diverse customer segments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1500
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
                
            campaign_strategy = json.loads(json_content.strip())
            return campaign_strategy
        except Exception as e:
            # If parsing fails, return the raw text
            return {
                "error": str(e),
                "raw_strategy": response.choices[0].message.content
            }
