from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.auth import Principal, get_current_principal, require_user
from app.services.auth_service import create_access_token, verify_password, hash_password

router = APIRouter(prefix="/auth", tags=["auth"])


async def _get_session():
    from app.main import async_session_maker
    async with async_session_maker() as session:
        yield session


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str
    name: str
    role: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(_get_session)):
    row = (await db.execute(
        text("SELECT id, email, name, role, password_hash, is_active FROM users WHERE email = :email"),
        {"email": body.email.lower().strip()},
    )).fetchone()

    if not row or not row.is_active or not verify_password(body.password, row.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(user_id=row.id, role=row.role)
    return LoginResponse(
        access_token=token,
        user_id=row.id,
        email=row.email,
        name=row.name,
        role=row.role,
    )


@router.get("/me")
async def me(principal: Principal = Depends(require_user)):
    if principal.is_api_key:
        return {"user_id": None, "role": "user", "is_api_key": True}
    return {
        "user_id": principal.user_id,
        "email": principal.email,
        "name": principal.name,
        "role": principal.role,
    }


@router.post("/change-password", status_code=204)
async def change_password(
    body: ChangePasswordRequest,
    principal: Principal = Depends(require_user),
    db: AsyncSession = Depends(_get_session),
):
    if principal.is_api_key:
        raise HTTPException(status_code=403, detail="API keys cannot change passwords")

    row = (await db.execute(
        text("SELECT password_hash FROM users WHERE id = :id"),
        {"id": principal.user_id},
    )).fetchone()

    if not row or not verify_password(body.current_password, row.password_hash):
        raise HTTPException(status_code=401, detail="Current password incorrect")

    await db.execute(
        text("UPDATE users SET password_hash = :h, updated_at = NOW() WHERE id = :id"),
        {"h": hash_password(body.new_password), "id": principal.user_id},
    )
    await db.commit()
