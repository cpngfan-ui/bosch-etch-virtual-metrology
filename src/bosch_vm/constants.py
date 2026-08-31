"""Dataset constants for the public BOSCH plasma-etch experiment.

The anonymous gas-channel names and sensor engineering units are not mapped in
the public files.  The constants below therefore preserve the raw names and use
``short``/``long`` phase terminology instead of assigning gas chemistry.
"""

from __future__ import annotations

from typing import NamedTuple

CHANNEL_PREFIX = "Stat3_Etch_MV_"

COMMON_PROCESS_CHANNELS: tuple[str, ...] = (
    f"{CHANNEL_PREFIX}EpdIntensity",
    f"{CHANNEL_PREFIX}ForeLinePressure",
    f"{CHANNEL_PREFIX}Gas1Flow",
    f"{CHANNEL_PREFIX}Gas2Flow",
    f"{CHANNEL_PREFIX}Gas3Flow",
    f"{CHANNEL_PREFIX}Gas4Flow",
    f"{CHANNEL_PREFIX}Gas5Flow",
    f"{CHANNEL_PREFIX}Gas7Flow",
    f"{CHANNEL_PREFIX}Gas8Flow",
    f"{CHANNEL_PREFIX}Heater1Temp",
    f"{CHANNEL_PREFIX}Heater2Temp",
    f"{CHANNEL_PREFIX}Heater3Temp",
    f"{CHANNEL_PREFIX}Heater4Temp",
    f"{CHANNEL_PREFIX}HeliumBPFlow",
    f"{CHANNEL_PREFIX}HeliumBPPressure",
    f"{CHANNEL_PREFIX}PlatenDcBias",
    f"{CHANNEL_PREFIX}PlatenRFLoadCapacitor",
    f"{CHANNEL_PREFIX}PlatenRFLoadPower",
    f"{CHANNEL_PREFIX}PlatenRFPeakToPeak",
    f"{CHANNEL_PREFIX}PlatenRFReflectedPower",
    f"{CHANNEL_PREFIX}PlatenRFTuningCapacitor",
    f"{CHANNEL_PREFIX}Pressure",
    f"{CHANNEL_PREFIX}SourceRF2LoadPower",
    f"{CHANNEL_PREFIX}SourceRF2PeakToPeak",
    f"{CHANNEL_PREFIX}SourceRF2ReflectedPower",
    f"{CHANNEL_PREFIX}SourceRF2TuningCapacitor",
    f"{CHANNEL_PREFIX}SourceRFLoadPower",
    f"{CHANNEL_PREFIX}SourceRFPeakToPeak",
    f"{CHANNEL_PREFIX}SourceRFReflectedPower",
    f"{CHANNEL_PREFIX}SourceRFTuningCapacitor",
    f"{CHANNEL_PREFIX}moriInnerCurrent",
)

SHORT_PHASE_PROXY = f"{CHANNEL_PREFIX}Gas4Flow"
LONG_PHASE_PROXY = f"{CHANNEL_PREFIX}Gas5Flow"
ACTIVE_WINDOW_PROXY = f"{CHANNEL_PREFIX}SourceRFLoadPower"

# Four target-free summaries are generated for each channel: overall level,
# phase contrast, phase-contrast MAD, and late-minus-early drift.
CYCLE_SUMMARY_CHANNELS: tuple[str, ...] = (
    f"{CHANNEL_PREFIX}Gas4Flow",
    f"{CHANNEL_PREFIX}Gas5Flow",
    f"{CHANNEL_PREFIX}ForeLinePressure",
    f"{CHANNEL_PREFIX}Pressure",
    f"{CHANNEL_PREFIX}HeliumBPFlow",
    f"{CHANNEL_PREFIX}PlatenDcBias",
    f"{CHANNEL_PREFIX}PlatenRFLoadPower",
    f"{CHANNEL_PREFIX}PlatenRFPeakToPeak",
    f"{CHANNEL_PREFIX}PlatenRFReflectedPower",
    f"{CHANNEL_PREFIX}PlatenRFLoadCapacitor",
    f"{CHANNEL_PREFIX}PlatenRFTuningCapacitor",
    f"{CHANNEL_PREFIX}SourceRFLoadPower",
    f"{CHANNEL_PREFIX}SourceRFPeakToPeak",
    f"{CHANNEL_PREFIX}SourceRFReflectedPower",
)

# Two run-level summaries are generated for each slow-state channel.
SLOW_STATE_CHANNELS: tuple[str, ...] = (
    f"{CHANNEL_PREFIX}Heater2Temp",
    f"{CHANNEL_PREFIX}Heater3Temp",
    f"{CHANNEL_PREFIX}Heater4Temp",
    f"{CHANNEL_PREFIX}Gas2Flow",
    f"{CHANNEL_PREFIX}Gas7Flow",
)

CONSTANT_ZERO_CHANNELS: tuple[str, ...] = (
    f"{CHANNEL_PREFIX}Gas3Flow",
    f"{CHANNEL_PREFIX}Gas8Flow",
    f"{CHANNEL_PREFIX}SourceRF2LoadPower",
    f"{CHANNEL_PREFIX}SourceRF2ReflectedPower",
)

EXPECTED_CYCLE_COUNT = 100
STABLE_CYCLE_START = 1  # zero-based cycle 2
STABLE_CYCLE_STOP = 99  # exclusive; zero-based cycle 99 is included
EXPECTED_PROCESS_RUNS = 96
EXPECTED_MATCHED_RUNS = 88
EXPECTED_SPATIAL_SITES = 89

CYCLE_FEATURES_PER_CHANNEL = 4
TIMING_FEATURE_COUNT = 10
SLOW_FEATURES_PER_CHANNEL = 2
CONTEXT_FEATURE_COUNT = 5
EXPECTED_PREDICTOR_COUNT = (
    len(CYCLE_SUMMARY_CHANNELS) * CYCLE_FEATURES_PER_CHANNEL
    + TIMING_FEATURE_COUNT
    + len(SLOW_STATE_CHANNELS) * SLOW_FEATURES_PER_CHANNEL
    + CONTEXT_FEATURE_COUNT
)


class LotDefinition(NamedTuple):
    """Context shared by all wafer runs from one experiment date."""

    lot_number: int
    conditioning_count: int
    conditioning_surface: str


LOT_BY_DATE: dict[str, LotDefinition] = {
    "2024-07-02": LotDefinition(1, 3, "chuck"),
    "2024-07-05": LotDefinition(2, 1, "chuck"),
    "2024-07-09": LotDefinition(3, 9, "chuck"),
    "2024-07-11": LotDefinition(4, 3, "si"),
    "2024-07-19": LotDefinition(5, 1, "si"),
    "2024-08-01": LotDefinition(6, 9, "si"),
    "2024-08-05": LotDefinition(7, 3, "sio2"),
    "2024-08-07": LotDefinition(8, 3, "sio2"),
    "2024-08-21": LotDefinition(9, 3, "chuck"),
    "2024-08-22": LotDefinition(10, 3, "sio2"),
}

CONDITIONING_SURFACES: tuple[str, ...] = ("chuck", "si", "sio2")

PROCESS_DATA_FILENAME = "Process_data.nc"
PROCESS_DICTIONARY_FILENAME = "Dictionary_process.nc"
METROLOGY_89_FILENAME = "Si_Oxide_etch_89_points.csv"
