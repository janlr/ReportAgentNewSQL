from typing import Dict, Any, List, Optional, Union
import pandas as pd
import json
import logging
from pathlib import Path
import hashlib
from datetime import datetime
import sqlite3
from .base_agent import BaseAgent

class DataManagerAgent(BaseAgent):
    """Agent responsible for data operations and favorites management."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("data_manager_agent")
        self.config = config
        
        # Set up directories
        self.data_dir = Path(config.get("data_dir", "./data"))
        self.favorites_dir = self.data_dir / "favorites"
        self.imports_dir = self.data_dir / "imports"
        
        for directory in [self.data_dir, self.favorites_dir, self.imports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for favorites
        self.db_path = self.data_dir / "favorites.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for favorites."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create favorites table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS favorites (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    report_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            
            # Create tags table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    favorite_id TEXT,
                    tag TEXT,
                    PRIMARY KEY (favorite_id, tag),
                    FOREIGN KEY (favorite_id) REFERENCES favorites(id)
                        ON DELETE CASCADE
                )
            """)
            
            conn.commit()
    
    async def initialize(self) -> bool:
        """Initialize the data manager."""
        try:
            self.logger.info("Initializing data manager")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing data manager: {str(e)}")
            return False
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data management requests."""
        action = input_data.get("action")
        
        if not action:
            raise ValueError("Action not specified in input data")
        
        try:
            if action == "import_excel":
                file_path = input_data.get("file_path")
                sheet_name = input_data.get("sheet_name")
                return await self._import_excel(file_path, sheet_name)
            
            elif action == "add_favorite":
                return await self._add_favorite(input_data)
            
            elif action == "get_favorite":
                favorite_id = input_data.get("favorite_id")
                return await self._get_favorite(favorite_id)
            
            elif action == "list_favorites":
                report_type = input_data.get("report_type")
                tags = input_data.get("tags")
                return await self._list_favorites(report_type, tags)
            
            elif action == "delete_favorite":
                favorite_id = input_data.get("favorite_id")
                return await self._delete_favorite(favorite_id)
            
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error processing data request: {str(e)}")
            raise
    
    async def cleanup(self):
        """Clean up data manager resources."""
        self.logger.info("Cleaning up data manager resources")
    
    async def _import_excel(self, file_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """Import data from an Excel file."""
        try:
            # Read Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            # Generate unique file hash
            file_hash = self._generate_file_hash(file_path)
            
            # Save as parquet for efficient storage
            output_path = self.imports_dir / f"{file_hash}.parquet"
            df.to_parquet(output_path, index=False)
            
            # Save metadata
            metadata = {
                "original_file": file_path,
                "sheet_name": sheet_name,
                "columns": list(df.columns),
                "row_count": len(df),
                "imported_at": datetime.now().isoformat()
            }
            
            metadata_path = self.imports_dir / f"{file_hash}_meta.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Imported Excel file {file_path} ({len(df)} rows)")
            return {
                "file_hash": file_hash,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error importing Excel file: {str(e)}")
            raise
    
    async def _add_favorite(self, input_data: Dict[str, Any]) -> str:
        """Add a report to favorites."""
        name = input_data.get("name")
        if not name:
            raise ValueError("Name is required for favorite")
        
        # Generate unique ID
        favorite_id = self._generate_favorite_id(name)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert favorite
                cursor.execute("""
                    INSERT INTO favorites (
                        id, name, description, report_type, parameters
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    favorite_id,
                    name,
                    input_data.get("description"),
                    input_data.get("report_type"),
                    json.dumps(input_data.get("parameters", {}))
                ))
                
                # Insert tags
                tags = set(input_data.get("tags", []))  # Remove duplicates
                for tag in tags:
                    cursor.execute("""
                        INSERT INTO tags (favorite_id, tag)
                        VALUES (?, ?)
                    """, (favorite_id, tag))
                
                conn.commit()
            
            self.logger.info(f"Added favorite {favorite_id}")
            return favorite_id
            
        except Exception as e:
            self.logger.error(f"Error adding favorite: {str(e)}")
            raise
    
    async def _get_favorite(self, favorite_id: str) -> Dict[str, Any]:
        """Get a favorite by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get favorite
                cursor.execute("""
                    SELECT 
                        id, name, description, report_type, parameters,
                        created_at, updated_at, usage_count
                    FROM favorites
                    WHERE id = ?
                """, (favorite_id,))
                
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Favorite {favorite_id} not found")
                
                # Get tags
                cursor.execute("""
                    SELECT tag
                    FROM tags
                    WHERE favorite_id = ?
                """, (favorite_id,))
                
                tags = [tag[0] for tag in cursor.fetchall()]
                
                # Update usage count
                cursor.execute("""
                    UPDATE favorites
                    SET usage_count = usage_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (favorite_id,))
                
                conn.commit()
                
                return {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "report_type": row[3],
                    "parameters": json.loads(row[4]),
                    "created_at": row[5],
                    "updated_at": row[6],
                    "usage_count": row[7],
                    "tags": tags
                }
                
        except Exception as e:
            self.logger.error(f"Error getting favorite: {str(e)}")
            raise
    
    async def _list_favorites(self, report_type: Optional[str] = None,
                            tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List favorites with optional filtering."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = """
                    SELECT DISTINCT
                        f.id, f.name, f.description, f.report_type,
                        f.parameters, f.created_at, f.updated_at, f.usage_count
                    FROM favorites f
                """
                
                params = []
                where_clauses = []
                
                if report_type:
                    where_clauses.append("f.report_type = ?")
                    params.append(report_type)
                
                if tags:
                    query += """
                        INNER JOIN tags t ON f.id = t.favorite_id
                        WHERE t.tag IN ({})
                    """.format(",".join("?" * len(tags)))
                    params.extend(tags)
                    
                    # Count matching tags
                    query += """
                        GROUP BY f.id
                        HAVING COUNT(DISTINCT t.tag) = ?
                    """
                    params.append(len(tags))
                
                elif where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
                
                query += " ORDER BY f.usage_count DESC"
                
                # Execute query
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Get tags for each favorite
                favorites = []
                for row in rows:
                    cursor.execute("""
                        SELECT tag
                        FROM tags
                        WHERE favorite_id = ?
                    """, (row[0],))
                    
                    tags = [tag[0] for tag in cursor.fetchall()]
                    
                    favorites.append({
                        "id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "report_type": row[3],
                        "parameters": json.loads(row[4]),
                        "created_at": row[5],
                        "updated_at": row[6],
                        "usage_count": row[7],
                        "tags": tags
                    })
                
                return favorites
                
        except Exception as e:
            self.logger.error(f"Error listing favorites: {str(e)}")
            raise
    
    async def _delete_favorite(self, favorite_id: str):
        """Delete a favorite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete favorite (cascade will delete tags)
                cursor.execute("""
                    DELETE FROM favorites
                    WHERE id = ?
                """, (favorite_id,))
                
                if cursor.rowcount == 0:
                    raise ValueError(f"Favorite {favorite_id} not found")
                
                conn.commit()
            
            self.logger.info(f"Deleted favorite {favorite_id}")
            
        except Exception as e:
            self.logger.error(f"Error deleting favorite: {str(e)}")
            raise
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Generate a unique hash for a file."""
        components = [
            Path(file_path).name,
            str(Path(file_path).stat().st_size),
            datetime.now().strftime("%Y%m%d")
        ]
        return hashlib.md5("_".join(components).encode()).hexdigest()[:12]
    
    def _generate_favorite_id(self, name: str) -> str:
        """Generate a unique favorite ID."""
        components = [
            name.lower().replace(" ", "_"),
            datetime.now().strftime("%Y%m%d%H%M%S")
        ]
        return hashlib.md5("_".join(components).encode()).hexdigest()[:12] 