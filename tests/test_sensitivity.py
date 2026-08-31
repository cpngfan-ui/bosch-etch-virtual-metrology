"""Small contract tests for the robustness suite."""

from bosch_vm.sensitivity import SENSITIVITY_VARIANTS, feature_block


def test_sensitivity_variant_ids_are_unique_and_include_measurement_lineage() -> None:
    identifiers = [variant.variant_id for variant in SENSITIVITY_VARIANTS]
    assert len(identifiers) == len(set(identifiers))
    assert "direct_postox_target" in identifiers
    assert "robust_scaler" in identifiers


def test_feature_block_mapping_is_physical_and_context_separated() -> None:
    assert feature_block("context__wafer_order") == "context"
    assert feature_block("cycle__Gas4Flow__level_median") == "gas_delivery"
    assert feature_block("cycle__ForeLinePressure__phase_contrast_median") == (
        "pressure_helium"
    )
    assert feature_block("cycle__SourceRFLoadPower__late_minus_early") == "rf_power"
    assert feature_block("slow__Heater2Temp__active_median") == "temperature"
    assert feature_block("timing__cycle_period_median_s") == "timing"
