## Live Demo

[Try Crop Recommendation System](https://crop-recommendation-using-soil-analysis-1.onrender.com)

# Crop Recommendation Using Soil Analysis

**A machine-learning system that recommends the most suitable crop for a given set of soil and climatic parameters.**

[Source Code](https://github.com/himnad/Crop-Recommendation-using-soil-analysis)

## Overview

Agriculture remains a primary-sector contributor to the economy, yet the continued use of identical cultivation methods and fertilizers leads to nutrient imbalance and declining soil health. This project applies data analysis and machine learning to soil fertility and climatic parameters in order to support evidence-based crop selection.

The system integrates exploratory data analysis, feature engineering, and supervised classification with principles of soil science. Its objective is to improve decision-making in the agricultural sector and to support environmental sustainability.

## Problem Statement

The persistent use of traditional cultivation practices produces the following consequences:

| Issue | Effect |
|---|---|
| Repeated use of the same methods and fertilizers | Nutrient imbalance and weakened soil |
| Declining soil health | Reduced crop yields and poor crop growth |
| Weakened crops | Higher vulnerability to pests and diseases |
| Cumulative degradation | Loss of soil fertility, quality, and resilience |

## Input Parameters

The model accepts the following soil and climatic parameters:

| Parameter | Unit (as presented in the project) | Example value |
|---|---|---|
| Nitrogen (N) | kg/ha | 23 |
| Phosphorus (P) | kg/ha | 50 |
| Potassium (K) | kg/ha | 180 |
| Temperature | °C | 30 |
| Humidity | gm/kg | 14 |
| Rainfall | cm per year | 100 |
| pH | — | 5 |

## Methodology

1. **Data collection:** Soil samples are gathered from areas of high crop productivity.
2. **Soil analysis and exploratory data analysis:** Fertility parameters are analyzed for specific crops, including the correlation between parameters and the distributions of pH, rainfall, and nitrogen.
3. **Data preprocessing:** Data is encoded and standardized in preparation for model training.
4. **Model training:** Several classifiers are trained on soil and climatic data, including Logistic Regression, Decision Tree, Support Vector Machine, and Random Forest.
5. **Crop recommendation:** The Random Forest classifier is selected on the basis of accuracy and is used to predict the best-suited crop for new soil data.

## Demonstration

For the example input listed above (N = 23 kg/ha, P = 50 kg/ha, K = 180 kg/ha, temperature = 30 °C, humidity = 14 gm/kg, rainfall = 100 cm per year, pH = 5), the system recommends:

> **Recommended crop: Banana**

## Repository Contents

| File / Folder | Description |
|---|---|
| `Crop_recommendation.csv` | Dataset of soil and climatic parameters with crop labels |
| `Projectcrop_recomendd.ipynb` | Notebook containing the exploratory analysis, preprocessing, and model training |
| `rf_classifier.pkl` | Serialized, trained Random Forest classifier |
| `std.pkl` | Serialized preprocessing object used to scale input data |
| `capp.py` | Application script for serving predictions |
| `templates/` | HTML templates used by the application |
| `Soil Analysis & Crop Recommendation.pptx` | Project presentation |
| `tempCodeRunnerFile.py` | Temporary editor-generated file; it may be removed |

## Getting Started

### Prerequisites

- Python 3.8 or later
- The libraries used by the project (for example, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, and a web framework for `capp.py`)

### Installation and Execution

```bash
git clone https://github.com/himnad/Crop-Recommendation-using-soil-analysis.git
cd Crop-Recommendation-using-soil-analysis
pip install numpy pandas scikit-learn matplotlib seaborn flask
python capp.py
```

Open the local address displayed in the terminal and enter the soil and climatic parameters to obtain a crop recommendation.

> **Note:** The dependency list and launch command above are inferred from the repository layout and should be verified against `capp.py` and the notebook. A `requirements.txt` file is recommended.

## Challenges

- Data quality and accessibility
- Environmental dynamics
- User adoption and trust
- Resource constraints
- Long-term sustainability

## Future Scope

- Wider global agricultural impact
- Increased crop yields
- Deeper soil health analysis
- Optimized utilization of resources
- Improved economic sustainability for farmers

## Conclusion

By combining soil science with machine learning, this project provides a data-driven means of selecting crops that suit local soil and climatic conditions, thereby promoting higher productivity, efficient use of resources, and sustainable agricultural practice.

