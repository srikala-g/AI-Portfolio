#!/usr/bin/env python3
"""
Test script to verify dataset loading functionality
"""

import os
import sys
from app import MalariaDataset

def test_dataset_loading():
    """Test the dataset loading functionality."""
    print("Testing dataset loading...")
    
    # Test with local path first
    dataset_handler = MalariaDataset(data_path='cell_images/')
    dataset = dataset_handler.load_dataset()
    
    if dataset is not None:
        print("✓ Dataset loaded successfully")
        
        # Try to get a sample
        try:
            sample = next(iter(dataset.take(1)))
            print(f"✓ Sample loaded: {sample.keys()}")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Label: {sample['label']}")
        except Exception as e:
            print(f"✗ Error getting sample: {e}")
    else:
        print("✗ Failed to load dataset")

if __name__ == "__main__":
    test_dataset_loading()

