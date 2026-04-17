from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.profile import (
    BehavioralAdjProfile,
    BrainChemistryProfile,
    EnvironmentDetailsProfile,
    FinalProfile,
    TypeDetailProfile,
)


PROFILE_ROWS = [
    {
        "user_id_hash": "user_1",
        "structure": 0.9,
        "answer_first": 0.95,
        "tone_directness": 0.8,
        "detail_level": 0.35,
        "ambiguity_reduction": 0.8,
        "exploration_level": 0.2,
        "context_loading": 0.3,
        "prompt_enforcement_level": "none",
        "compliance_check_enabled": False,
        "pii_check_enabled": False,
        "profile_version": "summary_type_1",
    },
    {
        "user_id_hash": "user_2",
        "structure": 0.85,
        "answer_first": 0.75,
        "tone_directness": 0.7,
        "detail_level": 0.55,
        "ambiguity_reduction": 0.8,
        "exploration_level": 0.3,
        "context_loading": 0.4,
        "prompt_enforcement_level": "none",
        "compliance_check_enabled": False,
        "pii_check_enabled": False,
        "profile_version": "summary_type_2",
    },
    {
        "user_id_hash": "user_3",
        "structure": 0.75,
        "answer_first": 0.7,
        "tone_directness": 0.65,
        "detail_level": 0.75,
        "ambiguity_reduction": 0.85,
        "exploration_level": 0.35,
        "context_loading": 0.55,
        "prompt_enforcement_level": "none",
        "compliance_check_enabled": False,
        "pii_check_enabled": False,
        "profile_version": "summary_type_3",
    },
    {
        "user_id_hash": "user_4",
        "structure": 0.6,
        "answer_first": 0.65,
        "tone_directness": 0.5,
        "detail_level": 0.8,
        "ambiguity_reduction": 0.75,
        "exploration_level": 0.45,
        "context_loading": 0.7,
        "prompt_enforcement_level": "none",
        "compliance_check_enabled": False,
        "pii_check_enabled": False,
        "profile_version": "summary_type_4",
    },
    {
        "user_id_hash": "user_5",
        "structure": 0.45,
        "answer_first": 0.4,
        "tone_directness": 0.45,
        "detail_level": 0.6,
        "ambiguity_reduction": 0.55,
        "exploration_level": 0.75,
        "context_loading": 0.7,
        "prompt_enforcement_level": "none",
        "compliance_check_enabled": False,
        "pii_check_enabled": False,
        "profile_version": "summary_type_5",
    },
    {
        "user_id_hash": "user_6",
        "structure": 0.35,
        "answer_first": 0.35,
        "tone_directness": 0.4,
        "detail_level": 0.45,
        "ambiguity_reduction": 0.4,
        "exploration_level": 0.9,
        "context_loading": 0.8,
        "prompt_enforcement_level": "none",
        "compliance_check_enabled": False,
        "pii_check_enabled": False,
        "profile_version": "summary_type_6",
    },
    {
        "user_id_hash": "user_7",
        "structure": 0.7,
        "answer_first": 0.8,
        "tone_directness": 0.85,
        "detail_level": 0.5,
        "ambiguity_reduction": 0.9,
        "exploration_level": 0.2,
        "context_loading": 0.45,
        "prompt_enforcement_level": "none",
        "compliance_check_enabled": False,
        "pii_check_enabled": False,
        "profile_version": "summary_type_7",
    },
    {
        "user_id_hash": "user_8",
        "structure": 0.55,
        "answer_first": 0.55,
        "tone_directness": 0.35,
        "detail_level": 0.85,
        "ambiguity_reduction": 0.65,
        "exploration_level": 0.55,
        "context_loading": 0.8,
        "prompt_enforcement_level": "none",
        "compliance_check_enabled": False,
        "pii_check_enabled": False,
        "profile_version": "summary_type_8",
    },
    {
        "user_id_hash": "user_9",
        "structure": 0.25,
        "answer_first": 0.25,
        "tone_directness": 0.3,
        "detail_level": 0.3,
        "ambiguity_reduction": 0.35,
        "exploration_level": 0.85,
        "context_loading": 0.35,
        "prompt_enforcement_level": "none",
        "compliance_check_enabled": False,
        "pii_check_enabled": False,
        "profile_version": "summary_type_9",
    },
]


def seed_table(session: Session, model: type[FinalProfile]) -> None:
    for row in PROFILE_ROWS:
        exists = session.get(model, row["user_id_hash"])
        if exists:
            for key, value in row.items():
                setattr(exists, key, value)
            continue
        session.add(model(**row))


def run_seed() -> None:
    session = SessionLocal()
    try:
        for model in (
            FinalProfile,
            TypeDetailProfile,
            BrainChemistryProfile,
            EnvironmentDetailsProfile,
            BehavioralAdjProfile,
        ):
            seed_table(session, model)
        session.commit()
    finally:
        session.close()


if __name__ == "__main__":
    run_seed()
