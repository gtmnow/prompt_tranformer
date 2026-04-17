from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ProfileMixin:
    user_id_hash: Mapped[str] = mapped_column(String(255), primary_key=True)
    structure: Mapped[float] = mapped_column(Float, nullable=False)
    answer_first: Mapped[float] = mapped_column(Float, nullable=False)
    tone_directness: Mapped[float] = mapped_column(Float, nullable=False)
    detail_level: Mapped[float] = mapped_column(Float, nullable=False)
    ambiguity_reduction: Mapped[float] = mapped_column(Float, nullable=False)
    exploration_level: Mapped[float] = mapped_column(Float, nullable=False)
    context_loading: Mapped[float] = mapped_column(Float, nullable=False)
    prompt_enforcement_level: Mapped[str] = mapped_column(String(20), nullable=False, default="none")
    compliance_check_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    pii_check_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    profile_version: Mapped[str] = mapped_column(String(50), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class FinalProfile(ProfileMixin, Base):
    __tablename__ = "final_profile"


class TypeDetailProfile(ProfileMixin, Base):
    __tablename__ = "type_detail"


class BrainChemistryProfile(ProfileMixin, Base):
    __tablename__ = "brain_chemistry"


class EnvironmentDetailsProfile(ProfileMixin, Base):
    __tablename__ = "environment_details"


class BehavioralAdjProfile(ProfileMixin, Base):
    __tablename__ = "behaviorial_adj"
