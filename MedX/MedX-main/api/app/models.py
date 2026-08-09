"""SQLAlchemy models mirroring db/schema.sql.

The schema is the source of truth — constraints live in Postgres, not here. These
classes are a typed access layer over it, deliberately not a second place where
invariants are declared.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint, Date, DateTime, ForeignKey, Integer, Numeric, String,
    Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    phone: Mapped[str | None] = mapped_column(String, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False, default="buyer")
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    email_verified: Mapped[bool] = mapped_column(default=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    seller: Mapped[Seller | None] = relationship(back_populates="user", uselist=False)


class Seller(Base):
    __tablename__ = "sellers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False
    )
    business_name: Mapped[str] = mapped_column(Text, nullable=False)
    # Form 20/21 retail/wholesale drug licence. A seller cannot list until this
    # is verified — legal requirement, not a trust score.
    license_number: Mapped[str] = mapped_column(Text, nullable=False)
    license_expiry: Mapped[date] = mapped_column(Date, nullable=False)
    gstin: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    address_line: Mapped[str] = mapped_column(Text, nullable=False)
    city: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    pincode: Mapped[str] = mapped_column(Text, nullable=False)
    region_code: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[float] = mapped_column(Numeric(3, 2), default=0)
    rating_count: Mapped[int] = mapped_column(Integer, default=0)

    user: Mapped[User] = relationship(back_populates="seller")
    listings: Mapped[list[Listing]] = relationship(back_populates="seller")

    @property
    def can_list(self) -> bool:
        return self.status == "verified" and self.license_expiry > date.today()


class Drug(Base):
    __tablename__ = "drugs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    brand_name: Mapped[str] = mapped_column(Text, nullable=False)
    composition: Mapped[str] = mapped_column(Text, nullable=False)
    # Normalized join key shared by every brand of the same molecule. Generic
    # substitute search depends on this.
    composition_key: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    strength: Mapped[str] = mapped_column(Text, nullable=False)
    form: Mapped[str] = mapped_column(String, nullable=False)
    manufacturer: Mapped[str] = mapped_column(Text, nullable=False)
    mrp: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    schedule_class: Mapped[str] = mapped_column(String, nullable=False, default="OTC")
    atc_code: Mapped[str | None] = mapped_column(Text, index=True)
    pack_size: Mapped[int] = mapped_column(Integer, default=1)

    @property
    def requires_prescription(self) -> bool:
        return self.schedule_class in ("H", "H1", "X")


class Listing(Base):
    """One lot: a specific batch of a specific drug from a specific seller.

    Keyed per batch rather than per product because expiry is intrinsic to a lot
    and because a recall must be executable as "disable every lot with batch X".
    """

    __tablename__ = "listings"
    __table_args__ = (
        UniqueConstraint("seller_id", "drug_id", "batch_number"),
        CheckConstraint("quantity_available <= quantity_total"),
        CheckConstraint("current_price <= mrp"),
        CheckConstraint("current_price >= price_floor"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    seller_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sellers.id"), nullable=False
    )
    drug_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("drugs.id"), nullable=False
    )
    batch_number: Mapped[str] = mapped_column(Text, nullable=False)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=False)
    quantity_total: Mapped[int] = mapped_column(Integer, nullable=False)
    quantity_available: Mapped[int] = mapped_column(Integer, nullable=False)
    mrp: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    current_price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    # Automated repricing may never go below this.
    price_floor: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    listed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    seller: Mapped[Seller] = relationship(back_populates="listings")
    drug: Mapped[Drug] = relationship()

    @property
    def days_to_expiry(self) -> int:
        return max((self.expiry_date - date.today()).days, 0)

    @property
    def discount_pct(self) -> float:
        return round((1 - float(self.current_price) / float(self.mrp)) * 100, 1)


class LotRiskScore(Base):
    __tablename__ = "lot_risk_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    listing_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("listings.id"), nullable=False
    )
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    days_to_expiry: Mapped[int] = mapped_column(Integer, nullable=False)
    p_sellout: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    expected_units_p10: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    expected_units_p50: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    expected_units_p90: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    recommended_price: Mapped[float | None] = mapped_column(Numeric(10, 2))
    tier: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    seller_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sellers.id"), nullable=False
    )
    listing_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("listings.id"), nullable=False
    )
    tier: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditLog(Base):
    """Append-only. No UPDATE/DELETE grant is issued in any environment."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id")
    )
    action: Mapped[str] = mapped_column(Text, nullable=False)
    entity_type: Mapped[str] = mapped_column(Text, nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    before: Mapped[dict | None] = mapped_column(JSONB)
    after: Mapped[dict | None] = mapped_column(JSONB)
    ip_address: Mapped[str | None] = mapped_column(INET)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
