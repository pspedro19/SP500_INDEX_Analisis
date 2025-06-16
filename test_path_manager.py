#!/usr/bin/env python3
"""Test script for the refactored PathManager and InvestingProcessor."""

import sys
import os
from pathlib import Path

# Add the pipelines path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipelines', 'ml', 'utils'))

from path_manager import PathManager
from sp500_analysis.config.settings import settings

def test_path_manager():
    """Test PathManager functionality."""
    print("=" * 60)
    print("TESTING PATHMANAGER")
    print("=" * 60)
    
    # Initialize PathManager with correct path
    data_root = settings.project_root / "data" / "raw" / "0_raw"
    print(f"Data root: {data_root}")
    
    path_manager = PathManager(data_root)
    
    # Test basic functionality
    print("\n1. Cache Statistics:")
    stats = path_manager.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n2. Available Categories:")
    categories = path_manager.get_available_categories()
    for cat in categories:
        print(f"   - {cat}")
    
    print("\n3. Testing file search:")
    test_files = [
        "US_ISM_Manufacturing.xlsx",
        "US_Consumer_Confidence.xlsx", 
        "China_PMI_Manufacturing.xlsx",
        "nonexistent_file.xlsx"
    ]
    
    for filename in test_files:
        result = path_manager.find_file(filename)
        if result:
            print(f"   [FOUND] {filename} -> {result}")
        else:
            print(f"   [NOT FOUND] {filename}")
            suggestions = path_manager.suggest_similar_files(filename, max_suggestions=3)
            if suggestions:
                print(f"     Similar: {', '.join(suggestions)}")
    
    print("\n4. File validation test:")
    required_files = [
        "US_ISM_Manufacturing.xlsx",
        "US_ISM_Services.xlsx", 
        "China_PMI_Manufacturing.xlsx",
        "nonexistent_file.xlsx"
    ]
    
    validation_results = path_manager.validate_required_files(required_files)
    found = sum(validation_results.values())
    total = len(validation_results)
    print(f"   Found: {found}/{total} files")
    
    return path_manager

def test_investing_processor():
    """Test the refactored InvestingProcessor."""
    print("\n" + "=" * 60)
    print("TESTING REFACTORED INVESTING PROCESSOR")
    print("=" * 60)
    
    try:
        from sp500_analysis.application.preprocessing.processors.investing import InvestingProcessor
        
        config_file = settings.project_root / "data" / "Data Engineering.xlsx"
        data_root = settings.project_root / "data" / "raw" / "0_raw"
        
        print(f"Config file: {config_file}")
        print(f"Data root: {data_root}")
        
        # Initialize processor
        processor = InvestingProcessor(
            config_file=str(config_file),
            data_root=str(data_root),
            log_file="test_investing_processor.log"
        )
        
        print("\nTesting configuration loading...")
        config_result = processor.read_config()
        if config_result is not None:
            print(f"[SUCCESS] Configuration loaded: {len(config_result)} entries")
        else:
            print("[FAILED] Failed to load configuration")
            return False
        
        print("\nTesting file validation...")
        validation_result = processor.validate_required_files()
        if validation_result:
            print("[SUCCESS] File validation passed")
        else:
            print("[FAILED] File validation failed")
        
        print("\nPathManager cache statistics:")
        stats = processor.path_manager.get_cache_stats()
        print(f"   Total files: {stats['total_files']}")
        print(f"   Categories: {stats['total_categories']}")
        print(f"   Scan duration: {stats['scan_duration_seconds']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing InvestingProcessor: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("Starting PathManager and InvestingProcessor tests...")
    
    # Test PathManager
    path_manager = test_path_manager()
    
    # Test InvestingProcessor
    processor_test_result = test_investing_processor()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"PathManager: ✓ Working")
    print(f"InvestingProcessor: {'✓ Working' if processor_test_result else '✗ Failed'}")
    
    # Export file inventory
    if path_manager:
        try:
            inventory_file = "file_inventory_test.xlsx"
            if path_manager.export_inventory(inventory_file):
                print(f"File inventory exported to: {inventory_file}")
        except Exception as e:
            print(f"Could not export inventory: {e}")

if __name__ == "__main__":
    main() 