"""Sanity checks on config.yaml (structure only, no network)."""
import math

EXPECTED_TICKERS = {"ZM", "MSFT", "MDB", "ESTC", "NVDA"}
EXPECTED_FORCES = [
    "Competitive_Rivalry", "Threat_of_New_Entry", "Threat_of_Substitutes",
    "Buyer_Power", "Supplier_Power", "Complementors",
]


def test_seed_is_42(config):
    assert config["random_seed"] == 42


def test_sec_settings(config):
    assert isinstance(config["sec_user_agent"], str) and config["sec_user_agent"]
    assert config["sec_request_sleep_s"] >= 0.1  # < 10 req/s


def test_tickers_and_no_hardcoded_cik(config):
    tickers = {c["ticker"] for c in config["companies"]}
    assert tickers == EXPECTED_TICKERS
    assert all("cik" not in c for c in config["companies"])


def test_forces(config):
    assert config["forces"] == EXPECTED_FORCES
    assert set(config["baseline"]["prototypes"]) == set(EXPECTED_FORCES)


def test_weights_sum_to_one(config):
    w = config["scoring"]["weights"]
    assert set(w) == set(EXPECTED_FORCES)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)


def test_anchors_present_and_distinct(config):
    anchors = config["scoring"]["anchors"]
    for name in ("mention_share", "gross_margin", "dol", "lerner_trend"):
        a = anchors[name]
        assert a["value_at_1"] != a["value_at_10"], name


def test_anchor_basis_marked(config):
    for name, a in config["scoring"]["anchors"].items():
        assert a["basis_at_1"] in {"SOURCED", "JUDGEMENT"}, name
        assert a["basis_at_10"] in {"SOURCED", "JUDGEMENT"}, name
        assert a.get("note"), name


def test_prototypes_filled(config):
    protos = config["baseline"]["prototypes"]
    for force in EXPECTED_FORCES:
        assert len(protos[force]) >= 3, force
        assert all(isinstance(s, str) and s.strip() for s in protos[force])


def test_label_guide_exists(config):
    from pathlib import Path
    assert (Path(__file__).resolve().parents[1] / config["labelling"]["label_guide"]).is_file()
