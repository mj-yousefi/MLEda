import unittest
import pandas as pd
from mleda.eda import EDA

class TestMLEda(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Sample DataFrame for testing
        cls.df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'feature_with_nan': [5, 4, None, 2, 1],
            'target': [1, 0, 1, 0, 1]
        })
        cls.eda = EDA(cls.df)
    
    def test_summary_statistics(self):
        """Test the summary statistics method"""
        summary = self.eda.summary_statistics()
        self.assertTrue('feature1' in summary.columns)
        self.assertTrue('feature2' in summary.columns)
        self.assertTrue('target' in summary.columns)
        self.assertFalse('invalid_feature' in summary.columns)

    def test_missing_values_with_none(self):
        """Test the missing values method"""
        missing = self.eda.missing_values()
        print('missing 1', missing)
        self.assertTrue((missing == 1).any())  # No missing values in test data

    def test_missing_values_whitout_none(self):
        """Test the missing values method"""
        missing = self.eda.missing_values(col_subset=['feature1', 'feature2', 'target'])
        print('missing 2', missing)
        self.assertTrue((missing == 0).all())  # No missing values in test data

    def test_correlation_heatmap(self):
        """Test the correlation heatmap method (visual check)"""
        # This is a visual test; ensure no errors are raised when displaying the heatmap
        try:
            self.eda.correlation_heatmap()
        except Exception as e:
            self.fail(f"correlation_heatmap() raised an exception: {e}")

    def test_target_variable_analysis(self):
        """Test the target variable analysis method (visual check)"""
        # This is a visual test; ensure no errors are raised
        try:
            self.eda.target_variable_analysis('target')
        except Exception as e:
            self.fail(f"target_variable_analysis() raised an exception: {e}")

    def test_correlation_with_target(self):
        """Test the correlation with target method"""
        correlation = self.eda.correlation_with_target('target')
        # Check if the result is a Series
        self.assertTrue(isinstance(correlation, pd.Series))

        # Verify that the Series index contains the columns from the DataFrame
        for column in self.df.columns:
            if column != 'target':  # Exclude target itself
                self.assertIn(column, correlation.index)
        
        # Optionally, check that the values are between -1 and 1 (range for correlation coefficients)
        self.assertTrue(((correlation >= -1) & (correlation <= 1)).all())

    def test_outlier_analysis(self):
        """Test the outlier analysis method (visual check)"""
        # This is a visual test; ensure no errors are raised
        try:
            self.eda.outlier_analysis(['feature1', 'feature2'])
        except Exception as e:
            self.fail(f"outlier_analysis() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
