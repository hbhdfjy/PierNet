from __future__ import annotations

from piern.simulators.naming import scenario_name_from_output


def test_scenario_name_from_output_strips_hdf5_suffix_and_simulator_prefix():
    assert scenario_name_from_output("data/simpeg", "simpeg_external.hdf5") == "external"
    assert scenario_name_from_output("data/modflow", "modflow_coastal.h5") == "coastal"


def test_scenario_name_from_output_preserves_unprefixed_stem():
    assert scenario_name_from_output("data/modflow", "external.hdf5") == "external"
    assert scenario_name_from_output("", "standalone.h5") == "standalone"
