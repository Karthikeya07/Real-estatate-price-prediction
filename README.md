# Real Estate Price Prediction System

A comprehensive machine learning project for predicting real estate prices in Bangalore, India, featuring multiple ML algorithms and an intelligent property recommendation system.

## Overview

This project predicts property prices based on location, size (square footage), number of bedrooms (BHK), and bathrooms. It compares seven different machine learning algorithms to select the best-performing model and provides property recommendations within user-specified budgets.

## Features

- **Multi-Algorithm Comparison**: Tests 7 different ML algorithms to find the best performer
  - Linear Regression
  - Decision Tree Regressor
  - MLP Regressor (Neural Network)
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
  - K-Nearest Neighbors Regressor

- **Comprehensive Data Preprocessing**:
  - Handles missing values and data type conversions
  - Feature engineering (BHK extraction, price per sqft calculation)
  - Outlier removal using statistical methods
  - Location categorization for rare locations

- **Intelligent Property Recommendations**:
  - Predicts property prices based on user requirements
  - Finds properties within specified area range
  - Identifies ready-to-move properties within predicted price
  - Detects overpriced properties
  - Suggests affordable locations within budget

- **One-Hot Encoding**: Efficient location encoding for better model performance

## Dataset

The dataset contains real estate listings from Bangalore with the following features:
- Location
- Size (BHK)
- Total square footage
- Number of bathrooms
- Balcony information
- Area type
- Availability status
- Society name
- Price (in Lakhs)

## Project Structure

```
Real-estatate-price-prediction/
│
├── real_estate_price_prediction.ipynb   # Main Jupyter notebook with complete analysis
├── price_prediction.py                  # Python script version
├── bangalore_real_estate_dataset.csv    # Dataset
├── project_report.pdf                   # Project report
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Karthikeya07/Real-estatate-price-prediction.git
cd Real-estatate-price-prediction
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Jupyter Notebook

```bash
jupyter notebook real_estate_price_prediction.ipynb
```

### Running the Python Script

```bash
python price_prediction.py
```

### Making Predictions

The system will prompt you for:
1. **Location**: Enter the desired location in Bangalore
2. **Square Footage**: Enter the required area in square feet
3. **Bathrooms**: Number of bathrooms needed
4. **Bedrooms (BHK)**: Number of bedrooms required
5. **Flexible Area Range**: Acceptable range variation in square footage
6. **Budget**: Maximum budget in INR

## Model Performance

The project automatically selects the best-performing algorithm based on cross-validation scores. Each model is evaluated using:
- Training/Test split (80/20)
- 5-fold cross-validation
- Mean squared error metrics

## Key Functionalities

### 1. Price Prediction
Predicts property price based on location, size, and specifications.

### 2. Property Filtering
- Lists all properties matching exact specifications
- Filters ready-to-move properties within predicted price range

### 3. Overpriced Property Detection
Identifies properties priced higher than the predicted market value.

### 4. Budget-Based Location Search
Finds all affordable locations within your budget for specified requirements.

### 5. Complete Property Recommendations
Returns all available properties matching specifications within budget across affordable locations.

## Data Preprocessing Steps

1. **Null Value Handling**: Removes rows with missing critical data
2. **Feature Engineering**:
   - Extracts BHK from size column
   - Converts square footage ranges to averages
   - Calculates price per square foot
3. **Outlier Removal**:
   - Removes properties with abnormally low sqft/BHK ratio (<300)
   - Removes price outliers using standard deviation method
4. **Location Encoding**:
   - Consolidates rare locations (< 10 properties) into "other" category
   - Applies one-hot encoding for model input

## Technologies Used

- **Python 3.x**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Keras, TensorFlow
- **Visualization**: Matplotlib, Seaborn
- **Development Environment**: Jupyter Notebook

## Results

The model successfully:
- Compares multiple algorithms and selects the best performer
- Achieves high accuracy in price prediction
- Provides practical property recommendations
- Handles real-world data inconsistencies

## Future Enhancements

- Web application interface for easier user interaction
- Integration with live property listings
- Additional features (property age, amenities, connectivity)
- Deep learning models for improved accuracy
- Visualization dashboard for market trends

## Author

**Karthikeya Nagandla**

## License

This project is available for educational and portfolio purposes.

## Acknowledgments

- Dataset sourced from Bangalore real estate listings
- Built as a machine learning demonstration project
- Inspired by real-world property price prediction needs
