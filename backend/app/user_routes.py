# app/user_routes.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_db
from .schemas import UserResponse, UserUpdate, ChangePasswordRequest
from .auth import get_current_active_user, require_admin
from .models import User
from .user_service import UserService

router = APIRouter(tags=["users"])

@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get all users (admin only)."""
    user_service = UserService(db)
    users = await user_service.get_all_users(skip=skip, limit=limit)
    return users

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get user by ID (admin only)."""
    user_service = UserService(db)
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Update user (admin only)."""
    user_service = UserService(db)
    user = await user_service.update_user(user_id, user_data)
    return user

@router.post("/{user_id}/change-password")
async def change_password(
    user_id: int,
    password_data: ChangePasswordRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Change user password (own password or admin)."""
    # Users can only change their own password unless they're admin
    if current_user.id != user_id and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    user_service = UserService(db)
    success = await user_service.change_password(
        user_id, 
        password_data.current_password, 
        password_data.new_password
    )
    return {"message": "Password changed successfully"}

@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Delete user (admin only - soft delete)."""
    user_service = UserService(db)
    success = await user_service.delete_user(user_id)
    return {"message": "User deactivated successfully"}
