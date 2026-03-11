from pipeline.run_daily import build_meta_signals


def test_build_meta_signals_respects_weights_and_votes():
    signals = {
        "brazil_soy_north": {"trading_action": "FLAT", "confidence": "Low", "signal": "No trade"},
        "brazil_soy_central": {"trading_action": "LONG", "confidence": "Medium", "signal": "Long crop"},
        "brazil_soy_southeast": {"trading_action": "SHORT", "confidence": "High", "signal": "Short crop"},
    }
    region_configs = {
        "brazil_soy_north": {"meta_group": "brazil_soy", "meta_weight": 0.35},
        "brazil_soy_central": {"meta_group": "brazil_soy", "meta_weight": 0.40},
        "brazil_soy_southeast": {"meta_group": "brazil_soy", "meta_weight": 0.25},
    }

    meta_signals = build_meta_signals(signals, region_configs)
    meta = meta_signals["brazil_soy_meta"]

    assert meta["trading_action"] == "FLAT"
    assert meta["confidence"] == "Low"
    assert len(meta["constituents"]) == 3


def test_build_meta_signals_can_turn_long():
    signals = {
        "brazil_soy_north": {"trading_action": "LONG", "confidence": "High", "signal": "Long crop"},
        "brazil_soy_central": {"trading_action": "LONG", "confidence": "High", "signal": "Long crop"},
        "brazil_soy_southeast": {"trading_action": "SHORT", "confidence": "Low", "signal": "Short crop"},
    }
    region_configs = {
        "brazil_soy_north": {"meta_group": "brazil_soy", "meta_weight": 0.35},
        "brazil_soy_central": {"meta_group": "brazil_soy", "meta_weight": 0.40},
        "brazil_soy_southeast": {"meta_group": "brazil_soy", "meta_weight": 0.25},
    }

    meta_signals = build_meta_signals(signals, region_configs)
    meta = meta_signals["brazil_soy_meta"]

    assert meta["trading_action"] == "LONG"
    assert meta["actionability"] == "Actionable"
    assert meta["vote_score"] > 0.2
