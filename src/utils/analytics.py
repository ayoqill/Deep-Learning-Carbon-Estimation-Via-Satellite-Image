# filepath: src/utils/analytics.py
# Analytics storage and management for analysis results

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

class AnalyticsManager:
    """
    Manages storage and retrieval of analysis results.
    Uses JSON file for simplicity. Can be upgraded to database later.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_file = self.storage_path / "analytics.json"
        self._ensure_db()
    
    def _ensure_db(self):
        """Ensure analytics.json exists"""
        if not self.db_file.exists():
            with open(self.db_file, 'w') as f:
                json.dump({"records": []}, f, indent=2)
    
    def _load_db(self) -> Dict[str, Any]:
        """Load analytics database"""
        with open(self.db_file, 'r') as f:
            return json.load(f)
    
    def _save_db(self, data: Dict[str, Any]):
        """Save analytics database"""
        with open(self.db_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Save an analysis result.
        
        Args:
            analysis: Dict with keys:
                - type: "uploaded" or "precomputed"
                - title: str (filename or label)
                - location: str (optional, e.g., "Langkawi")
                - originalImagePath: str (relative or absolute)
                - resultImagePath: str (relative or absolute)
                - model: str (e.g., "unetpp", "deeplabv3")
                - mangroveCoverage: float (%)
                - totalAreaHectares: float
                - totalAreaM2: float
                - carbonStock: float (tons)
                - co2Equivalent: float (tons)
                - pixelSizeM: float
        
        Returns:
            str: analysis id
        """
        db = self._load_db()
        
        # Generate unique ID
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Add metadata
        analysis['id'] = analysis_id
        analysis['createdAt'] = datetime.now().isoformat()
        
        db['records'].append(analysis)
        self._save_db(db)
        
        return analysis_id
    
    def get_all_analyses(self) -> List[Dict[str, Any]]:
        """Get all analysis records, sorted by date (newest first)"""
        db = self._load_db()
        records = db.get('records', [])
        # Sort by createdAt descending
        records.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        return records
    
    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific analysis by ID"""
        db = self._load_db()
        for record in db.get('records', []):
            if record.get('id') == analysis_id:
                return record
        return None
    
    def get_by_type(self, analysis_type: str) -> List[Dict[str, Any]]:
        """Get all analyses of a specific type (uploaded/precomputed)"""
        db = self._load_db()
        records = [r for r in db.get('records', []) if r.get('type') == analysis_type]
        records.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        return records
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete an analysis by ID"""
        db = self._load_db()
        original_count = len(db['records'])
        db['records'] = [r for r in db['records'] if r.get('id') != analysis_id]
        
        if len(db['records']) < original_count:
            self._save_db(db)
            return True
        return False
    
    def clear_all(self):
        """Clear all records (use with caution)"""
        self._save_db({"records": []})
