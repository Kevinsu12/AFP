"""Data loading utilities for parquet files."""

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_parquet_data(
    data_dir: str,
    split: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Load parquet data from the data directory.
    
    Args:
        data_dir: Path to data directory
        split: Data split ('train', 'test', 'val') or specific filename
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        Combined DataFrame from all parquet files in the split
    """
    data_path = Path(data_dir)
    
    # Check if split is a specific file or a directory
    if split.endswith('.parquet'):
        # Direct file path
        file_path = data_path / split
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file {file_path} does not exist")
        parquet_files = [file_path]
    else:
        # Directory-based approach (original behavior)
        split_path = data_path / split
        if not split_path.exists():
            raise FileNotFoundError(f"Data directory {split_path} does not exist")
        
        # Find all parquet files in the directory
        parquet_files = list(split_path.glob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {split_path}")
    
    # Load the parquet file
    df = pd.read_parquet(parquet_files[0])
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
        # Apply date filters if provided
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
    
    return df
