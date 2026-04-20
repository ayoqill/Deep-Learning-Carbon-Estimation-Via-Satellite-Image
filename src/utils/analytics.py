# filepath: src/utils/analytics.py
# Analytics storage and management for analysis results

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

class AnalyticsManager:
    """
    Manages storage and retrieval of analysis results.
    Supports per-user isolation of uploaded analyses.
    Precomputed study area analyses are shared across all users.
    Uses JSON file for simplicity. Can be upgraded to database later.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_file = self.storage_path / "analytics.json"
        self._ensure_db()
    
    def _ensure_db(self):
        """Ensure analytics.json exists with user structure"""
        if not self.db_file.exists():
            with open(self.db_file, 'w') as f:
                json.dump({
                    "users": {},  # {username: {records: [...]}}
                    "precomputed": []  # Shared across all users
                }, f, indent=2)
    
    def _load_db(self) -> Dict[str, Any]:
        """Load analytics database"""
        with open(self.db_file, 'r') as f:
            data = json.load(f)
            # Migrate legacy format if needed
            if "records" in data and "users" not in data:
                data = {
                    "users": {},
                    "precomputed": [r for r in data.get("records", []) if r.get("type") == "precomputed"],
                }
            return data
    
    def _save_db(self, data: Dict[str, Any]):
        """Save analytics database"""
        with open(self.db_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_analysis(self, analysis: Dict[str, Any], username: Optional[str] = None) -> str:
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
            username: str (optional, username of the uploader)
        
        Returns:
            str: analysis id
        """
        db = self._load_db()
        
        # Generate unique ID
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Add metadata
        analysis['id'] = analysis_id
        analysis['createdAt'] = datetime.now().isoformat()
        analysis['username'] = username
        
        # Route to correct location based on type
        if analysis.get('type') == 'precomputed':
            # Precomputed analyses are shared
            if 'precomputed' not in db:
                db['precomputed'] = []
            db['precomputed'].append(analysis)
        else:
            # Uploaded analyses are per-user
            if 'users' not in db:
                db['users'] = {}
            if username not in db['users']:
                db['users'][username] = {'records': []}
            db['users'][username]['records'].append(analysis)
        
        self._save_db(db)
        
        return analysis_id
    
    def get_all_analyses(self, username: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all analysis records.
        
        Args:
            username: If provided, returns only uploaded analyses for that user
                     If None, returns all precomputed analyses
        
        Returns:
            List of analysis records sorted by date (newest first)
        """
        db = self._load_db()
        records = []
        
        if username:
            # Get user's uploaded analyses
            if 'users' in db and username in db['users']:
                records = db['users'][username].get('records', [])
        else:
            # Get shared precomputed analyses
            records = db.get('precomputed', [])
        
        # Sort by createdAt descending
        records.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        return records
    
    def get_analysis(self, analysis_id: str, username: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific analysis by ID.
        
        Args:
            analysis_id: The analysis ID to retrieve
            username: If provided, ensures analysis belongs to this user (for uploaded)
        
        Returns:
            Analysis record or None if not found
        """
        db = self._load_db()
        
        if username:
            # Check user's uploaded analyses
            if 'users' in db and username in db['users']:
                for record in db['users'][username].get('records', []):
                    if record.get('id') == analysis_id:
                        return record
        else:
            # Check precomputed analyses
            for record in db.get('precomputed', []):
                if record.get('id') == analysis_id:
                    return record
        
        return None
    
    def get_by_type(self, analysis_type: str, username: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all analyses of a specific type.
        
        Args:
            analysis_type: "uploaded" or "precomputed"
            username: If provided and type is "uploaded", gets only that user's uploaded analyses
        
        Returns:
            List of matching records sorted by date (newest first)
        """
        db = self._load_db()
        records = []
        
        if analysis_type == 'uploaded' and username:
            # Get user's uploaded analyses
            if 'users' in db and username in db['users']:
                records = db['users'][username].get('records', [])
        elif analysis_type == 'precomputed':
            # Get shared precomputed analyses
            records = db.get('precomputed', [])
        
        records.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        return records
    
    def delete_analysis(self, analysis_id: str, username: Optional[str] = None) -> bool:
        """
        Delete an analysis by ID.
        
        Args:
            analysis_id: The analysis ID to delete
            username: If provided, ensures analysis belongs to this user
        
        Returns:
            True if deleted, False if not found
        """
        db = self._load_db()
        
        if username:
            # Delete from user's uploaded analyses
            if 'users' in db and username in db['users']:
                original_count = len(db['users'][username].get('records', []))
                db['users'][username]['records'] = [
                    r for r in db['users'][username]['records'] if r.get('id') != analysis_id
                ]
                if len(db['users'][username]['records']) < original_count:
                    self._save_db(db)
                    return True
        else:
            # Delete from precomputed
            original_count = len(db.get('precomputed', []))
            db['precomputed'] = [r for r in db.get('precomputed', []) if r.get('id') != analysis_id]
            if len(db.get('precomputed', [])) < original_count:
                self._save_db(db)
                return True
        
        return False
    
    def clear_all(self, username: Optional[str] = None):
        """
        Clear all records.
        
        Args:
            username: If provided, clears only that user's uploaded analyses
                     If None, clears all precomputed analyses
        """
        db = self._load_db()
        
        if username:
            # Clear user's uploaded analyses
            if 'users' not in db:
                db['users'] = {}
            db['users'][username] = {'records': []}
        else:
            # Clear precomputed analyses
            db['precomputed'] = []
        
        self._save_db(db)
