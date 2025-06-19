"""Efficient path management and file discovery for data processing pipelines."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd

from sp500_analysis.shared.logging.logger import configurar_logging


@dataclass
class FileInfo:
    """Information about a discovered file."""
    
    name: str
    full_path: str
    directory: str
    category: str
    size_bytes: int
    last_modified: float


class PathManager:
    """Efficient file path discovery and caching for data processing."""
    
    def __init__(self, data_root: str | Path, log_file: str = "path_manager.log"):
        """Initialize PathManager with data root directory.
        
        Parameters
        ----------
        data_root : str | Path
            Root directory containing data files
        log_file : str
            Log file for path manager operations
        """
        self.data_root = Path(data_root)
        self.logger = configurar_logging(log_file)
        
        # Cache structures
        self._file_cache: Dict[str, FileInfo] = {}
        self._category_cache: Dict[str, List[str]] = defaultdict(list)
        self._path_cache: Dict[str, str] = {}
        self._last_scan_time: Optional[float] = None
        self._scan_duration: Optional[float] = None
        
        # Configuration
        self.supported_extensions = {'.xlsx', '.csv', '.json', '.parquet'}
        self.cache_validity_seconds = 300  # 5 minutes
        
        self.logger.info(f"PathManager initialized for: {self.data_root}")
        
    def _is_cache_valid(self) -> bool:
        """Check if the current cache is still valid."""
        if self._last_scan_time is None:
            return False
        
        current_time = time.time()
        return (current_time - self._last_scan_time) < self.cache_validity_seconds
    
    def _scan_directory(self, force_rescan: bool = False) -> None:
        """Scan the data directory and build file cache.
        
        Parameters
        ----------
        force_rescan : bool
            Force a rescan even if cache is valid
        """
        if self._is_cache_valid() and not force_rescan:
            self.logger.debug("Using cached file information")
            return
            
        start_time = time.time()
        self.logger.info("Starting directory scan...")
        
        # Clear existing cache
        self._file_cache.clear()
        self._category_cache.clear()
        self._path_cache.clear()
        
        files_found = 0
        directories_scanned = 0
        
        if not self.data_root.exists():
            self.logger.error(f"Data root directory does not exist: {self.data_root}")
            return
            
        # Walk through all directories
        for root, dirs, files in os.walk(self.data_root):
            directories_scanned += 1
            root_path = Path(root)
            category = root_path.name if root_path != self.data_root else "root"
            
            for file in files:
                file_path = root_path / file
                file_ext = file_path.suffix.lower()
                
                # Only cache supported file types
                if file_ext in self.supported_extensions:
                    try:
                        stat_info = file_path.stat()
                        file_info = FileInfo(
                            name=file,
                            full_path=str(file_path),
                            directory=str(root_path),
                            category=category,
                            size_bytes=stat_info.st_size,
                            last_modified=stat_info.st_mtime
                        )
                        
                        # Store in different cache structures for fast lookups
                        file_key = file_path.stem  # filename without extension
                        self._file_cache[file_key] = file_info
                        self._category_cache[category].append(file_key)
                        self._path_cache[file_key] = str(file_path)
                        
                        files_found += 1
                        
                    except OSError as e:
                        self.logger.warning(f"Could not access file {file_path}: {e}")
        
        # Update scan metadata
        self._last_scan_time = time.time()
        self._scan_duration = self._last_scan_time - start_time
        
        self.logger.info(f"Directory scan completed:")
        self.logger.info(f"  - Files found: {files_found}")
        self.logger.info(f"  - Directories scanned: {directories_scanned}")
        self.logger.info(f"  - Categories found: {len(self._category_cache)}")
        self.logger.info(f"  - Scan duration: {self._scan_duration:.2f}s")
        
        # Log category breakdown
        for category, file_list in self._category_cache.items():
            self.logger.debug(f"  - {category}: {len(file_list)} files")
    
    def find_file(self, filename: str, force_rescan: bool = False) -> Optional[str]:
        """Find the full path of a file by name.
        
        Parameters
        ----------
        filename : str
            Name of the file to find (with or without extension)
        force_rescan : bool
            Force a directory rescan before searching
            
        Returns
        -------
        Optional[str]
            Full path to the file if found, None otherwise
        """
        self._scan_directory(force_rescan=force_rescan)
        
        # Remove extension if provided
        file_key = Path(filename).stem
        
        if file_key in self._path_cache:
            found_path = self._path_cache[file_key]
            self.logger.debug(f"File found: {filename} -> {found_path}")
            return found_path
        
        self.logger.warning(f"File not found: {filename}")
        return None
    
    def find_files_in_category(self, category: str, force_rescan: bool = False) -> List[str]:
        """Find all files in a specific category/directory.
        
        Parameters
        ----------
        category : str
            Category name (directory name)
        force_rescan : bool
            Force a directory rescan before searching
            
        Returns
        -------
        List[str]
            List of full paths to files in the category
        """
        self._scan_directory(force_rescan=force_rescan)
        
        if category not in self._category_cache:
            self.logger.warning(f"Category not found: {category}")
            return []
        
        file_keys = self._category_cache[category]
        paths = [self._path_cache[key] for key in file_keys if key in self._path_cache]
        
        self.logger.debug(f"Found {len(paths)} files in category '{category}'")
        return paths
    
    def validate_required_files(self, required_files: List[str]) -> Dict[str, bool]:
        """Validate which required files are available.
        
        Parameters
        ----------
        required_files : List[str]
            List of required filenames
            
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping filename to availability status
        """
        self._scan_directory()
        
        validation_results = {}
        found_count = 0
        
        for filename in required_files:
            file_key = Path(filename).stem
            is_available = file_key in self._path_cache
            validation_results[filename] = is_available
            
            if is_available:
                found_count += 1
                if self.logger:
                    self.logger.debug(f"[FOUND] {filename} -> {self._path_cache[file_key]}")
            else:
                if self.logger:
                    self.logger.warning(f"[NOT FOUND] {filename}")
        
        missing_count = len(required_files) - found_count
        
        self.logger.info(f"File validation summary:")
        self.logger.info(f"  - Required files: {len(required_files)}")
        self.logger.info(f"  - Found: {found_count}")
        self.logger.info(f"  - Missing: {missing_count}")
        
        if missing_count > 0:
            self.logger.warning(f"Missing files may cause processing errors")
        
        return validation_results
    
    def get_available_categories(self) -> List[str]:
        """Get list of all available data categories.
        
        Returns
        -------
        List[str]
            List of category names
        """
        self._scan_directory()
        categories = list(self._category_cache.keys())
        self.logger.debug(f"Available categories: {categories}")
        return categories
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get statistics about the current cache.
        
        Returns
        -------
        Dict[str, any]
            Cache statistics
        """
        return {
            'total_files': len(self._file_cache),
            'total_categories': len(self._category_cache),
            'last_scan_time': self._last_scan_time,
            'scan_duration_seconds': self._scan_duration,
            'cache_age_seconds': time.time() - self._last_scan_time if self._last_scan_time else None,
            'is_cache_valid': self._is_cache_valid(),
            'data_root': str(self.data_root)
        }
    
    def get_file_info(self, filename: str) -> Optional[FileInfo]:
        """Get detailed information about a file.
        
        Parameters
        ----------
        filename : str
            Name of the file
            
        Returns
        -------
        Optional[FileInfo]
            File information if found, None otherwise
        """
        self._scan_directory()
        file_key = Path(filename).stem
        return self._file_cache.get(file_key)
    
    def suggest_similar_files(self, filename: str, max_suggestions: int = 5) -> List[str]:
        """Suggest files with similar names.
        
        Parameters
        ----------
        filename : str
            Target filename
        max_suggestions : int
            Maximum number of suggestions to return
            
        Returns
        -------
        List[str]
            List of similar filenames
        """
        self._scan_directory()
        
        target = filename.lower()
        target_key = Path(filename).stem.lower()
        
        # Calculate similarity scores
        suggestions = []
        for file_key, file_info in self._file_cache.items():
            if file_key.lower() == target_key:
                continue  # Skip exact matches
                
            # Simple similarity based on common substrings
            similarity = self._calculate_similarity(target_key, file_key.lower())
            if similarity > 0.3:  # Threshold for suggestions
                suggestions.append((similarity, file_info.name))
        
        # Sort by similarity and return top suggestions
        suggestions.sort(reverse=True, key=lambda x: x[0])
        result = [name for _, name in suggestions[:max_suggestions]]
        
        if result:
            self.logger.info(f"Similar files found for '{filename}': {result}")
        
        return result
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity."""
        if not str1 or not str2:
            return 0.0
            
        # Simple Jaccard similarity on character bigrams
        bigrams1 = set(str1[i:i+2] for i in range(len(str1)-1))
        bigrams2 = set(str2[i:i+2] for i in range(len(str2)-1))
        
        if not bigrams1 and not bigrams2:
            return 1.0
        if not bigrams1 or not bigrams2:
            return 0.0
            
        intersection = len(bigrams1.intersection(bigrams2))
        union = len(bigrams1.union(bigrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def export_inventory(self, output_file: str) -> bool:
        """Export current file inventory to Excel.
        
        Parameters
        ----------
        output_file : str
            Output Excel file path
            
        Returns
        -------
        bool
            True if export successful, False otherwise
        """
        try:
            self._scan_directory()
            
            # Prepare data for export
            inventory_data = []
            for file_key, file_info in self._file_cache.items():
                inventory_data.append({
                    'filename': file_info.name,
                    'file_key': file_key,
                    'full_path': file_info.full_path,
                    'category': file_info.category,
                    'size_mb': file_info.size_bytes / (1024 * 1024),
                    'last_modified': pd.to_datetime(file_info.last_modified, unit='s')
                })
            
            # Create DataFrame and export
            df = pd.DataFrame(inventory_data)
            df = df.sort_values(['category', 'filename'])
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='File_Inventory', index=False)
                
                # Add summary sheet
                summary_data = []
                for category, files in self._category_cache.items():
                    total_size = sum(
                        self._file_cache[f].size_bytes 
                        for f in files 
                        if f in self._file_cache
                    ) / (1024 * 1024)
                    
                    summary_data.append({
                        'category': category,
                        'file_count': len(files),
                        'total_size_mb': total_size
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Category_Summary', index=False)
            
            self.logger.info(f"File inventory exported to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export inventory: {e}")
            return False