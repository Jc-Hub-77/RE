# backend/dependencies.py
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from .models import User
from .db import get_db
from .api.v1.auth_router import get_current_active_user # Import the dependency for active users

async def get_current_active_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """
    Dependency to get the current active admin user.
    Requires an authenticated and active user who also has the 'User.is_admin' flag set to True.
    This implementation directly relies on the `User.is_admin` boolean flag.
    Future enhancements might involve a more granular role-based access control (RBAC) system.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation requires admin privileges",
        )
    return current_user

# Add other common dependencies here if needed in the future
