from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.auth import Principal, require_admin, require_super_admin, require_user
from app.models.database import UserRoleEnum
from app.services.auth_service import hash_password

router = APIRouter(prefix="/users", tags=["users"])


async def _get_session():
    from app.main import async_session_maker
    async with async_session_maker() as session:
        yield session


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    role: str
    is_active: bool
    created_at: str
    updated_at: str


class CreateUserRequest(BaseModel):
    email: str
    name: str
    password: str
    role: str = "user"

    @field_validator("password")
    @classmethod
    def password_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in ("super_admin", "admin", "user"):
            raise ValueError("Role must be super_admin, admin, or user")
        return v


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[str] = None

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("super_admin", "admin", "user"):
            raise ValueError("Role must be super_admin, admin, or user")
        return v


def _row_to_out(row) -> dict:
    return {
        "id": row.id,
        "email": row.email,
        "name": row.name,
        "role": row.role,
        "is_active": bool(row.is_active),
        "created_at": row.created_at.isoformat() if row.created_at else "",
        "updated_at": row.updated_at.isoformat() if row.updated_at else "",
    }


@router.get("", response_model=List[UserOut])
async def list_users(
    principal: Principal = Depends(require_admin),
    db: AsyncSession = Depends(_get_session),
):
    rows = (await db.execute(
        text("SELECT id, email, name, role, is_active, created_at, updated_at FROM users ORDER BY created_at")
    )).fetchall()
    return [_row_to_out(r) for r in rows]


@router.post("", response_model=UserOut, status_code=201)
async def create_user(
    body: CreateUserRequest,
    principal: Principal = Depends(require_admin),
    db: AsyncSession = Depends(_get_session),
):
    # Admins can only create users (not admins or super_admins)
    if principal.role == UserRoleEnum.admin and body.role in ("admin", "super_admin"):
        raise HTTPException(status_code=403, detail="Admins can only create user-role accounts")

    existing = (await db.execute(
        text("SELECT id FROM users WHERE email = :email"),
        {"email": body.email.lower().strip()},
    )).fetchone()
    if existing:
        raise HTTPException(status_code=409, detail="Email already in use")

    now = datetime.now(timezone.utc)
    await db.execute(
        text("""INSERT INTO users (email, name, password_hash, role, is_active, created_by, created_at, updated_at)
                VALUES (:email, :name, :pw, :role, 1, :created_by, :now, :now)"""),
        {
            "email": body.email.lower().strip(),
            "name": body.name.strip(),
            "pw": hash_password(body.password),
            "role": body.role,
            "created_by": principal.user_id,
            "now": now,
        },
    )
    await db.commit()

    row = (await db.execute(
        text("SELECT id, email, name, role, is_active, created_at, updated_at FROM users WHERE email = :email"),
        {"email": body.email.lower().strip()},
    )).fetchone()
    return _row_to_out(row)


@router.get("/{user_id}", response_model=UserOut)
async def get_user(
    user_id: int,
    principal: Principal = Depends(require_admin),
    db: AsyncSession = Depends(_get_session),
):
    row = (await db.execute(
        text("SELECT id, email, name, role, is_active, created_at, updated_at FROM users WHERE id = :id"),
        {"id": user_id},
    )).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return _row_to_out(row)


@router.put("/{user_id}", response_model=UserOut)
async def update_user(
    user_id: int,
    body: UpdateUserRequest,
    principal: Principal = Depends(require_admin),
    db: AsyncSession = Depends(_get_session),
):
    row = (await db.execute(
        text("SELECT id, role FROM users WHERE id = :id"), {"id": user_id}
    )).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    # Admins cannot promote to admin/super_admin, nor edit existing admins
    if principal.role == UserRoleEnum.admin:
        if row.role in ("admin", "super_admin"):
            raise HTTPException(status_code=403, detail="Cannot modify admin or super_admin accounts")
        if body.role in ("admin", "super_admin"):
            raise HTTPException(status_code=403, detail="Cannot assign admin or super_admin role")

    # super_admins cannot demote themselves
    if user_id == principal.user_id and body.role and body.role != principal.role:
        raise HTTPException(status_code=400, detail="Cannot change your own role")

    updates, params = [], {"id": user_id, "now": datetime.now(timezone.utc)}
    if body.name is not None:
        updates.append("name = :name"); params["name"] = body.name.strip()
    if body.is_active is not None:
        updates.append("is_active = :active"); params["active"] = body.is_active
    if body.role is not None:
        updates.append("role = :role"); params["role"] = body.role

    if updates:
        await db.execute(
            text(f"UPDATE users SET {', '.join(updates)}, updated_at = :now WHERE id = :id"),
            params,
        )
        await db.commit()

    row = (await db.execute(
        text("SELECT id, email, name, role, is_active, created_at, updated_at FROM users WHERE id = :id"),
        {"id": user_id},
    )).fetchone()
    return _row_to_out(row)


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    principal: Principal = Depends(require_super_admin),
    db: AsyncSession = Depends(_get_session),
):
    if user_id == principal.user_id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    row = (await db.execute(
        text("SELECT id FROM users WHERE id = :id"), {"id": user_id}
    )).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    await db.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})
    await db.commit()


@router.post("/{user_id}/reset-password", status_code=204)
async def reset_password(
    user_id: int,
    body: dict,
    principal: Principal = Depends(require_admin),
    db: AsyncSession = Depends(_get_session),
):
    """Admin resets another user's password."""
    new_password = body.get("new_password", "")
    if len(new_password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")

    row = (await db.execute(
        text("SELECT role FROM users WHERE id = :id"), {"id": user_id}
    )).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    if principal.role == UserRoleEnum.admin and row.role in ("admin", "super_admin"):
        raise HTTPException(status_code=403, detail="Cannot reset admin passwords")

    await db.execute(
        text("UPDATE users SET password_hash = :h, updated_at = NOW() WHERE id = :id"),
        {"h": hash_password(new_password), "id": user_id},
    )
    await db.commit()
