from app.db.session import get_db
from app.models.profile import BrainChemistryProfile, FinalProfile, TypeDetailProfile
from app.services.profile_builder import ProfileBuilder


def _get_db(client):
    return next(client.app.dependency_overrides[get_db]())


def test_recompute_final_profile_uses_foundational_type_detail(client) -> None:
    db = _get_db(client)
    try:
        db.add(
            TypeDetailProfile(
                user_id_hash="user_foundational",
                structure=0.8,
                answer_first=0.7,
                tone_directness=0.6,
                detail_level=0.75,
                ambiguity_reduction=0.85,
                exploration_level=0.35,
                context_loading=0.55,
                prompt_enforcement_level="moderate",
                compliance_check_enabled=True,
                pii_check_enabled=False,
                profile_version="summary_type_1",
            )
        )
        db.commit()

        result = ProfileBuilder(db).recompute_final_profile("user_foundational")
        db.commit()

        final_profile = db.get(FinalProfile, "user_foundational")
        assert final_profile is not None
        assert result.applied_layers == ("type_detail",)
        assert result.profile_version == "summary_type_1"
        assert final_profile.structure == 0.8
        assert final_profile.prompt_enforcement_level == "moderate"
        assert final_profile.profile_version == "summary_type_1"
    finally:
        db.close()


def test_recompute_final_profile_overlays_higher_order_layers(client) -> None:
    db = _get_db(client)
    try:
        db.add(
            TypeDetailProfile(
                user_id_hash="user_layered",
                structure=0.8,
                answer_first=0.7,
                tone_directness=0.6,
                detail_level=0.75,
                ambiguity_reduction=0.85,
                exploration_level=0.35,
                context_loading=0.55,
                prompt_enforcement_level="none",
                compliance_check_enabled=False,
                pii_check_enabled=False,
                profile_version="summary_type_2",
            )
        )
        db.add(
            BrainChemistryProfile(
                user_id_hash="user_layered",
                structure=0.45,
                answer_first=0.4,
                tone_directness=0.5,
                detail_level=0.9,
                ambiguity_reduction=0.65,
                exploration_level=0.7,
                context_loading=0.8,
                prompt_enforcement_level="full",
                compliance_check_enabled=True,
                pii_check_enabled=True,
                profile_version="cqi_brain_profile_v1",
            )
        )
        db.commit()

        result = ProfileBuilder(db).recompute_final_profile("user_layered")
        db.commit()

        final_profile = db.get(FinalProfile, "user_layered")
        assert final_profile is not None
        assert result.applied_layers == ("type_detail", "brain_chemistry")
        assert result.profile_version == "summary_type_2+layers"
        assert final_profile.structure == 0.45
        assert final_profile.detail_level == 0.9
        assert final_profile.prompt_enforcement_level == "full"
        assert final_profile.compliance_check_enabled is True
        assert final_profile.pii_check_enabled is True
        assert final_profile.profile_version == "summary_type_2+layers"
    finally:
        db.close()
