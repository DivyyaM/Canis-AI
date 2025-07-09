"""
Role-Based Access Control (RBAC) System for Canis AI Backend
Provides team-scale access control for MLOps workflows
"""

import jwt
import bcrypt
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from functools import wraps
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

class Permission(Enum):
    """Available permissions in the system"""
    # Data permissions
    UPLOAD_DATA = "upload_data"
    VIEW_DATA = "view_data"
    DELETE_DATA = "delete_data"
    
    # Model permissions
    TRAIN_MODEL = "train_model"
    VIEW_MODEL = "view_model"
    DELETE_MODEL = "delete_model"
    DEPLOY_MODEL = "deploy_model"
    
    # Benchmark permissions
    RUN_BENCHMARK = "run_benchmark"
    VIEW_BENCHMARK = "view_benchmark"
    
    # Explainability permissions
    VIEW_EXPLANATIONS = "view_explanations"
    GENERATE_EXPLANATIONS = "generate_explanations"
    
    # Admin permissions
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    VIEW_LOGS = "view_logs"
    SYSTEM_CONFIG = "system_config"

class Role(Enum):
    """Predefined roles with their permissions"""
    ADMIN = "admin"
    ML_ENGINEER = "ml_engineer"
    DATA_SCIENTIST = "data_scientist"
    BUSINESS_USER = "business_user"
    VIEWER = "viewer"

# Role-permission mappings
ROLE_PERMISSIONS = {
    Role.ADMIN: [
        Permission.UPLOAD_DATA, Permission.VIEW_DATA, Permission.DELETE_DATA,
        Permission.TRAIN_MODEL, Permission.VIEW_MODEL, Permission.DELETE_MODEL, Permission.DEPLOY_MODEL,
        Permission.RUN_BENCHMARK, Permission.VIEW_BENCHMARK,
        Permission.VIEW_EXPLANATIONS, Permission.GENERATE_EXPLANATIONS,
        Permission.MANAGE_USERS, Permission.MANAGE_ROLES, Permission.VIEW_LOGS, Permission.SYSTEM_CONFIG
    ],
    Role.ML_ENGINEER: [
        Permission.UPLOAD_DATA, Permission.VIEW_DATA,
        Permission.TRAIN_MODEL, Permission.VIEW_MODEL, Permission.DELETE_MODEL, Permission.DEPLOY_MODEL,
        Permission.RUN_BENCHMARK, Permission.VIEW_BENCHMARK,
        Permission.VIEW_EXPLANATIONS, Permission.GENERATE_EXPLANATIONS,
        Permission.VIEW_LOGS
    ],
    Role.DATA_SCIENTIST: [
        Permission.UPLOAD_DATA, Permission.VIEW_DATA,
        Permission.TRAIN_MODEL, Permission.VIEW_MODEL,
        Permission.RUN_BENCHMARK, Permission.VIEW_BENCHMARK,
        Permission.VIEW_EXPLANATIONS, Permission.GENERATE_EXPLANATIONS
    ],
    Role.BUSINESS_USER: [
        Permission.VIEW_DATA, Permission.VIEW_MODEL,
        Permission.VIEW_BENCHMARK, Permission.VIEW_EXPLANATIONS
    ],
    Role.VIEWER: [
        Permission.VIEW_DATA, Permission.VIEW_MODEL, Permission.VIEW_BENCHMARK
    ]
}

class RBACManager:
    """Manages RBAC operations"""
    
    def __init__(self, db_path: str = "models/rbac.db"):
        self.db_path = db_path
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self._ensure_directories()
        self._init_database()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _init_database(self):
        """Initialize RBAC database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
            # Custom roles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS custom_roles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role_name TEXT UNIQUE NOT NULL,
                    permissions TEXT NOT NULL,
                    created_by INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (created_by) REFERENCES users (id)
                )
            ''')
            
            # User sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token_hash TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            # Create default admin user if not exists
            self._create_default_admin()
            
        except Exception as e:
            logger.error(f"Failed to initialize RBAC database: {str(e)}")
            raise
    
    def _create_default_admin(self):
        """Create default admin user"""
        try:
            if not self.get_user_by_username("admin"):
                self.create_user(
                    username="admin",
                    email="admin@canis.ai",
                    password="admin123",
                    role=Role.ADMIN.value
                )
                logger.info("Default admin user created")
        except Exception as e:
            logger.error(f"Failed to create default admin: {str(e)}")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_user(self, username: str, email: str, password: str, role: str) -> Dict[str, Any]:
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, role))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"User created: {username} with role {role}")
            return {"user_id": user_id, "username": username, "role": role}
            
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Username or email already exists")
        except Exception as e:
            logger.error(f"Failed to create user: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create user")
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user info"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, password_hash, role, is_active
                FROM users WHERE username = ?
            ''', (username,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            user_id, username, email, password_hash, role, is_active = row
            
            if not is_active:
                return None
            
            if not self.verify_password(password, password_hash):
                return None
            
            # Update last login
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
            conn.commit()
            conn.close()
            
            return {
                "user_id": user_id,
                "username": username,
                "email": email,
                "role": role
            }
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return None
    
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode = {
            "sub": str(user_data["user_id"]),
            "username": user_data["username"],
            "role": user_data["role"],
            "exp": expire
        }
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return user data"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def get_user_permissions(self, role: str) -> List[str]:
        """Get permissions for a role"""
        if role in ROLE_PERMISSIONS:
            return [perm.value for perm in ROLE_PERMISSIONS[Role(role)]]
        else:
            # Check custom roles
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT permissions FROM custom_roles WHERE role_name = ?
                ''', (role,))
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    import json
                    return json.loads(row[0])
                else:
                    return []
            except Exception as e:
                logger.error(f"Failed to get custom role permissions: {str(e)}")
                return []
    
    def has_permission(self, user_role: str, required_permission: str) -> bool:
        """Check if user has required permission"""
        user_permissions = self.get_user_permissions(user_role)
        return required_permission in user_permissions
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, email, role, is_active, created_at, last_login
                FROM users WHERE username = ?
            ''', (username,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "user_id": row[0],
                    "username": row[1],
                    "email": row[2],
                    "role": row[3],
                    "is_active": bool(row[4]),
                    "created_at": row[5],
                    "last_login": row[6]
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get user: {str(e)}")
            return None

# Global RBAC manager instance
rbac_manager = RBACManager()

def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from request context
            # This would need to be implemented based on your FastAPI setup
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            if not rbac_manager.has_permission(current_user["role"], permission.value):
                raise HTTPException(
                    status_code=403, 
                    detail=f"Permission denied: {permission.value} required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token"""
    token = credentials.credentials
    user_data = rbac_manager.verify_token(token)
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_data

def require_role(role: Role):
    """Decorator to require specific role"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            if current_user["role"] != role.value:
                raise HTTPException(
                    status_code=403,
                    detail=f"Role required: {role.value}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator 