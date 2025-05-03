# Smart Customer Segmentation and Data Analysis Platform

This project is an advanced customer segmentation and analysis platform powered by OpenAI's GPT-4 API. It combines machine learning algorithms with AI-driven insights to provide sophisticated customer segmentation and targeted marketing recommendations.

## Features

- 🔍 **Dynamic Customer Segmentation**
  - Automatic feature importance detection
  - Multiple clustering algorithms (K-Means, Hierarchical, DBSCAN)
  - Optimal segment count determination
  - Interactive visualization of segments

- 🤖 **AI-Powered Analysis**
  - Deep customer behavior analysis
  - Segment characteristic identification
  - Pattern recognition in customer data
  - GPT-4 powered insights generation

- 📊 **Advanced Data Processing**
  - Support for CSV, Excel, and JSON formats
  - Automated data preprocessing
  - Missing value handling
  - Feature scaling and encoding

- 💡 **Marketing Recommendations**
  - Segment-specific marketing strategies
  - Communication channel optimization
  - Campaign timing suggestions
  - Product recommendations per segment

- 📈 **Interactive Visualizations**
  - Segment distribution charts
  - Feature importance visualization
  - Segment profiles dashboard
  - Marketing recommendations interface

## Technology Stack

- **Backend:**
  - Python 3.8+
  - Flask (Web Server)
  - scikit-learn (Machine Learning)
  - pandas (Data Processing)
  - OpenAI API (GPT-4)

- **Frontend:**
  - HTML5
  - TailwindCSS
  - Chart.js (Visualizations)
  - XLSX.js (Excel Processing)
  - Papa Parse (CSV Processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Segment-Analysis-AI.git
cd Segment-Analysis-AI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your-api-key-here`

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload your customer data file (CSV, Excel, or JSON format)

4. Configure analysis settings:
   - Number of segments
   - Advanced analysis options
   - Marketing recommendations

5. View the results:
   - Segment distribution
   - Customer profiles
   - Marketing recommendations
   - Detailed analysis report

## Data Format Requirements

Your input data should contain customer-related information such as:
- Demographics
- Purchase history
- Behavioral data
- Engagement metrics
- Customer value indicators

Supported file formats:
- CSV files
- Excel files (.xlsx, .xls)
- JSON files

## Project Structure

```
Segment-Analysis-AI/
├── app.py                 # Flask application server
├── data_analyzer.py       # Data analysis and preprocessing
├── segment_analyzer.py    # Segment analysis and recommendations
├── segmentation_model.py  # ML models for segmentation
├── template/
│   └── index.html        # Web interface
└── requirements.txt      # Python dependencies
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- OpenAI for providing the GPT-4 API
- scikit-learn for machine learning algorithms
- TailwindCSS for the UI components
- Chart.js for data visualization