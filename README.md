# MLEda

MLEda is an automated Exploratory Data Analysis (EDA) tool designed specifically for machine learning projects. It helps data scientists and ML practitioners get quick insights into their data, understand relationships, and prepare data for modeling.

## Features
- Summary statistics and distributions of features
- Target variable analysis for classification and regression tasks
- Missing values analysis and handling recommendations
- Correlation matrix and feature relationships
- Initial feature importance scoring

## Installation
Install MLEda via pip:

```bash
pip install mleda
```


## Usage
```
import pandas as pd
from mleda import EDA

# Load your data
df = pd.read_csv('your_data.csv')

# Create an EDA instance
eda = EDA(df)

# Generate summary statistics
print(eda.summary_statistics())

# Check for missing values
print(eda.missing_values())

# Display a correlation heatmap
eda.correlation_heatmap()
```

## License

MIT License