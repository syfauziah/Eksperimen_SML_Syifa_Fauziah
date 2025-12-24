"""
automate_Syifa_Fauziah.py - Automated Wine Quality Data Preprocessing Pipeline

Author: Syifa Fauziah
Course: Membangun Sistem Machine Learning - Dicoding

This script automates the preprocessing pipeline for wine quality prediction,
converting the steps from Eksperimen_Syifa_Fauziah.ipynb into a production-ready script.

Usage:
    python automate_Syifa_Fauziah.py --source uci --output-dir winequality_preprocessing
    python automate_Syifa_Fauziah.py --source local --input-file data.csv --output-dir output
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)


class WineQualityPreprocessor:
    """
    Automated preprocessing pipeline for wine quality dataset.
    
    This class encapsulates all preprocessing steps including:
    - Data loading from UCI repository or local file
    - Exploratory data analysis metrics computation
    - Duplicate removal
    - Feature engineering
    - Categorical encoding
    - Outlier handling via IQR capping
    - Train-test splitting with stratification
    - Feature scaling using StandardScaler
    """
    
    RED_WINE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    WHITE_WINE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    
    FEATURE_COLUMNS = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'wine_type_encoded', 'total_acidity',
        'bound_sulfur_dioxide', 'sugar_to_alcohol', 'density_alcohol_ratio'
    ]
    
    OUTLIER_FEATURES = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'sulphates', 'total_acidity', 'bound_sulfur_dioxide', 'sugar_to_alcohol',
        'density_alcohol_ratio'
    ]
    
    def __init__(self, output_dir: str = 'winequality_preprocessing'):
        """
        Initialize the preprocessor.
        
        Args:
            output_dir: Directory to save preprocessed data and artifacts
        """
        self.output_dir = output_dir
        self.df_raw = None
        self.df_clean = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.le_wine_type = LabelEncoder()
        self.le_quality_category = LabelEncoder()
        self.stats = {}
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Output directory set to: {output_dir}')
    
    def load_data(self, source: str = 'uci', input_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load wine quality data from UCI repository or local file.
        
        Args:
            source: Data source ('uci' or 'local')
            input_file: Path to local CSV file (required if source='local')
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f'Loading data from source: {source}')
        
        if source == 'uci':
            df_red = pd.read_csv(self.RED_WINE_URL, sep=';')
            df_white = pd.read_csv(self.WHITE_WINE_URL, sep=';')
            
            df_red['wine_type'] = 'red'
            df_white['wine_type'] = 'white'
            
            self.df_raw = pd.concat([df_red, df_white], axis=0, ignore_index=True)
            
            logger.info(f'Red wine samples: {len(df_red)}')
            logger.info(f'White wine samples: {len(df_white)}')
            
        elif source == 'local':
            if input_file is None:
                raise ValueError('input_file must be provided when source is local')
            self.df_raw = pd.read_csv(input_file)
            
        else:
            raise ValueError(f'Unknown source: {source}. Use "uci" or "local"')
        
        logger.info(f'Total samples loaded: {len(self.df_raw)}')
        logger.info(f'Columns: {self.df_raw.columns.tolist()}')
        
        self.stats['total_samples_raw'] = len(self.df_raw)
        self.stats['n_columns_raw'] = len(self.df_raw.columns)
        
        return self.df_raw
    
    def compute_eda_metrics(self) -> Dict:
        """
        Compute exploratory data analysis metrics.
        
        Returns:
            Dictionary containing EDA statistics
        """
        logger.info('Computing EDA metrics')
        
        eda_metrics = {
            'shape': list(self.df_raw.shape),
            'missing_values': self.df_raw.isnull().sum().to_dict(),
            'duplicates': int(self.df_raw.duplicated().sum()),
            'dtypes': self.df_raw.dtypes.astype(str).to_dict(),
            'numeric_stats': {},
            'quality_distribution': self.df_raw['quality'].value_counts().sort_index().to_dict()
        }
        
        numeric_cols = self.df_raw.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            eda_metrics['numeric_stats'][col] = {
                'mean': float(self.df_raw[col].mean()),
                'std': float(self.df_raw[col].std()),
                'min': float(self.df_raw[col].min()),
                'max': float(self.df_raw[col].max()),
                'median': float(self.df_raw[col].median()),
                'q1': float(self.df_raw[col].quantile(0.25)),
                'q3': float(self.df_raw[col].quantile(0.75))
            }
        
        if 'wine_type' in self.df_raw.columns:
            eda_metrics['wine_type_distribution'] = self.df_raw['wine_type'].value_counts().to_dict()
        
        self.stats['eda_metrics'] = eda_metrics
        logger.info(f'Duplicates found: {eda_metrics["duplicates"]}')
        
        return eda_metrics
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Returns:
            Cleaned DataFrame without duplicates
        """
        initial_rows = len(self.df_raw)
        self.df_clean = self.df_raw.drop_duplicates()
        removed = initial_rows - len(self.df_clean)
        
        logger.info(f'Duplicates removed: {removed}')
        self.stats['duplicates_removed'] = removed
        self.stats['samples_after_dedup'] = len(self.df_clean)
        
        return self.df_clean
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Create new features from existing columns.
        
        Features created:
        - quality_category: Categorical version of quality (low/medium/high)
        - total_acidity: Sum of fixed and volatile acidity
        - bound_sulfur_dioxide: Total minus free sulfur dioxide
        - sugar_to_alcohol: Ratio of residual sugar to alcohol
        - density_alcohol_ratio: Ratio of density to alcohol
        
        Returns:
            DataFrame with engineered features
        """
        logger.info('Engineering new features')
        
        def categorize_quality(quality: int) -> str:
            if quality <= 4:
                return 'low'
            elif quality <= 6:
                return 'medium'
            return 'high'
        
        self.df_clean['quality_category'] = self.df_clean['quality'].apply(categorize_quality)
        
        self.df_clean['total_acidity'] = (
            self.df_clean['fixed acidity'] + self.df_clean['volatile acidity']
        )
        
        self.df_clean['bound_sulfur_dioxide'] = (
            self.df_clean['total sulfur dioxide'] - self.df_clean['free sulfur dioxide']
        )
        
        epsilon = 1e-6
        self.df_clean['sugar_to_alcohol'] = (
            self.df_clean['residual sugar'] / (self.df_clean['alcohol'] + epsilon)
        )
        
        self.df_clean['density_alcohol_ratio'] = (
            self.df_clean['density'] / (self.df_clean['alcohol'] + epsilon)
        )
        
        new_features = [
            'quality_category', 'total_acidity', 'bound_sulfur_dioxide',
            'sugar_to_alcohol', 'density_alcohol_ratio'
        ]
        logger.info(f'New features created: {new_features}')
        self.stats['engineered_features'] = new_features
        
        return self.df_clean
    
    def encode_categorical(self) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder.
        
        Returns:
            DataFrame with encoded categorical columns
        """
        logger.info('Encoding categorical variables')
        
        self.df_clean['wine_type_encoded'] = self.le_wine_type.fit_transform(
            self.df_clean['wine_type']
        )
        
        self.df_clean['quality_category_encoded'] = self.le_quality_category.fit_transform(
            self.df_clean['quality_category']
        )
        
        wine_type_mapping = dict(zip(
            self.le_wine_type.classes_,
            range(len(self.le_wine_type.classes_))
        ))
        quality_cat_mapping = dict(zip(
            self.le_quality_category.classes_,
            range(len(self.le_quality_category.classes_))
        ))
        
        logger.info(f'Wine type encoding: {wine_type_mapping}')
        logger.info(f'Quality category encoding: {quality_cat_mapping}')
        
        self.stats['encodings'] = {
            'wine_type': wine_type_mapping,
            'quality_category': quality_cat_mapping
        }
        
        return self.df_clean
    
    def cap_outliers(self) -> pd.DataFrame:
        """
        Cap outliers using the IQR method.
        
        Returns:
            DataFrame with capped outliers
        """
        logger.info('Capping outliers using IQR method')
        
        outlier_counts = {}
        
        for col in self.OUTLIER_FEATURES:
            if col not in self.df_clean.columns:
                continue
                
            Q1 = self.df_clean[col].quantile(0.25)
            Q3 = self.df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = (
                (self.df_clean[col] < lower_bound) | 
                (self.df_clean[col] > upper_bound)
            ).sum()
            
            self.df_clean[col] = self.df_clean[col].clip(
                lower=lower_bound, 
                upper=upper_bound
            )
            
            outlier_counts[col] = int(outliers_before)
        
        total_outliers = sum(outlier_counts.values())
        logger.info(f'Total outliers capped: {total_outliers}')
        self.stats['outliers_capped'] = outlier_counts
        
        return self.df_clean
    
    def split_data(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets with stratification.
        
        Args:
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f'Splitting data (test_size={test_size}, random_state={random_state})')
        
        X = self.df_clean[self.FEATURE_COLUMNS]
        y = self.df_clean['quality']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f'Training samples: {len(self.X_train)}')
        logger.info(f'Test samples: {len(self.X_test)}')
        
        self.stats['train_samples'] = len(self.X_train)
        self.stats['test_samples'] = len(self.X_test)
        self.stats['n_features'] = len(self.FEATURE_COLUMNS)
        self.stats['test_size'] = test_size
        self.stats['random_state'] = random_state
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler.
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        logger.info('Scaling features using StandardScaler')
        
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        self.X_train = pd.DataFrame(
            X_train_scaled, 
            columns=self.FEATURE_COLUMNS, 
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            X_test_scaled, 
            columns=self.FEATURE_COLUMNS, 
            index=self.X_test.index
        )
        
        logger.info('Feature scaling completed')
        
        return self.X_train, self.X_test
    
    def save_outputs(self) -> Dict[str, str]:
        """
        Save all preprocessed data and artifacts.
        
        Returns:
            Dictionary of saved file paths
        """
        logger.info(f'Saving outputs to {self.output_dir}')
        
        saved_files = {}
        
        self.X_train.to_csv(f'{self.output_dir}/X_train.csv', index=False)
        saved_files['X_train'] = f'{self.output_dir}/X_train.csv'
        
        self.X_test.to_csv(f'{self.output_dir}/X_test.csv', index=False)
        saved_files['X_test'] = f'{self.output_dir}/X_test.csv'
        
        self.y_train.to_csv(f'{self.output_dir}/y_train.csv', index=False, header=['quality'])
        saved_files['y_train'] = f'{self.output_dir}/y_train.csv'
        
        self.y_test.to_csv(f'{self.output_dir}/y_test.csv', index=False, header=['quality'])
        saved_files['y_test'] = f'{self.output_dir}/y_test.csv'
        
        self.df_clean.to_csv(
            f'{self.output_dir}/winequality_preprocessed_full.csv', 
            index=False
        )
        saved_files['full_data'] = f'{self.output_dir}/winequality_preprocessed_full.csv'
        
        joblib.dump(self.scaler, f'{self.output_dir}/scaler.pkl')
        saved_files['scaler'] = f'{self.output_dir}/scaler.pkl'
        
        joblib.dump(self.le_wine_type, f'{self.output_dir}/label_encoder_wine_type.pkl')
        saved_files['le_wine_type'] = f'{self.output_dir}/label_encoder_wine_type.pkl'
        
        joblib.dump(self.le_quality_category, f'{self.output_dir}/label_encoder_quality_category.pkl')
        saved_files['le_quality_cat'] = f'{self.output_dir}/label_encoder_quality_category.pkl'
        
        joblib.dump(self.FEATURE_COLUMNS, f'{self.output_dir}/feature_columns.pkl')
        saved_files['feature_columns'] = f'{self.output_dir}/feature_columns.pkl'
        
        self.stats['timestamp'] = datetime.now().isoformat()
        self.stats['feature_columns'] = self.FEATURE_COLUMNS
        
        with open(f'{self.output_dir}/preprocessing_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        saved_files['stats'] = f'{self.output_dir}/preprocessing_stats.json'
        
        logger.info(f'Saved {len(saved_files)} files')
        for name, path in saved_files.items():
            logger.info(f'  - {name}: {path}')
        
        return saved_files
    
    def run_pipeline(
        self, 
        source: str = 'uci', 
        input_file: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, str]:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            source: Data source ('uci' or 'local')
            input_file: Path to local file (if source='local')
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Dictionary of saved file paths
        """
        logger.info('='*60)
        logger.info('Starting Wine Quality Preprocessing Pipeline')
        logger.info('='*60)
        
        self.load_data(source=source, input_file=input_file)
        
        self.compute_eda_metrics()
        
        self.remove_duplicates()
        
        self.engineer_features()
        
        self.encode_categorical()
        
        self.cap_outliers()
        
        self.split_data(test_size=test_size, random_state=random_state)
        
        self.scale_features()
        
        saved_files = self.save_outputs()
        
        logger.info('='*60)
        logger.info('Pipeline completed successfully')
        logger.info('='*60)
        
        return saved_files


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Wine Quality Data Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python automate_Syifa_Fauziah.py --source uci --output-dir winequality_preprocessing
  python automate_Syifa_Fauziah.py --source local --input-file data.csv --output-dir output
  python automate_Syifa_Fauziah.py --source uci --test-size 0.3 --random-state 123
        '''
    )
    
    parser.add_argument(
        '--source',
        type=str,
        choices=['uci', 'local'],
        default='uci',
        help='Data source: "uci" for UCI repository or "local" for local file'
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Path to local CSV file (required if source is "local")'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='winequality_preprocessing',
        help='Directory to save preprocessed data and artifacts'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the preprocessing pipeline."""
    args = parse_arguments()
    
    if args.source == 'local' and args.input_file is None:
        logger.error('--input-file is required when --source is "local"')
        sys.exit(1)
    
    preprocessor = WineQualityPreprocessor(output_dir=args.output_dir)
    
    try:
        saved_files = preprocessor.run_pipeline(
            source=args.source,
            input_file=args.input_file,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        print('\nPreprocessing completed successfully!')
        print(f'Output directory: {args.output_dir}')
        print(f'Files saved: {len(saved_files)}')
        
        return 0
        
    except Exception as e:
        logger.error(f'Pipeline failed: {str(e)}')
        raise


if __name__ == '__main__':
    sys.exit(main())
