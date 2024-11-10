# mleda/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary_statistics(self):
        """Returns summary statistics of numerical features."""
        return self.df.describe()

    def missing_values(self, col_subset : list = None):
        """Returns a count of missing values in each column."""
        df = self.df.copy()
        if col_subset is not None and len(col_subset) > 0:
            df = df[col_subset]
        return df.isnull().sum()

    def correlation_heatmap(self):
        """Displays a heatmap of the correlation matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm")
        plt.show()

    def target_variable_analysis(self, target_col: str):
        """Analyzes the distribution of the target variable."""
        if self.df[target_col].dtype == 'object':
            # For classification tasks, show count plot
            sns.countplot(x=target_col, data=self.df)
        else:
            # For regression tasks, show distribution plot
            sns.histplot(self.df[target_col], kde=True)
        plt.title(f"Distribution of {target_col}")
        plt.show()

    def correlation_with_target(self, target_col: str):
        """Shows correlation of each feature with the target variable."""
        return self.df.corr()[target_col].sort_values(ascending=False)

    def outlier_analysis(self, columns=None):
        """Displays boxplots to visualize outliers for specified numerical columns."""
        if columns is None:
            columns = self.df.select_dtypes(include='number').columns
        
        for col in columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.df[col])
            plt.title(f"Outliers in {col}")
            plt.show()

