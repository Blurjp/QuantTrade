from paper_trading.multi_asset_portfolio import MultiAssetPortfolio
from pipeline.run_daily import apply_meta_signal_persistence, build_meta_signals, update_portfolio_with_signals


META_GROUPS = {
    "brazil_soy": {
        "label": "Brazil soy",
        "type": "meta_agriculture",
        "instruments": ["Soybeans"],
        "portfolio_trade": True,
        "confirmations_required": 2,
        "bullish_bias": "Bullish soybean prices",
        "bearish_bias": "Bearish soybean prices",
        "neutral_bias": "Mixed regional soybean signal",
    }
}


def test_build_meta_signals_respects_weights_and_votes(tmp_path):
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

    meta_signals = build_meta_signals(signals, region_configs, META_GROUPS, str(tmp_path))
    meta = meta_signals["brazil_soy_meta"]

    assert meta["trading_action"] == "FLAT"
    assert meta["confidence"] == "Low"
    assert len(meta["constituents"]) == 3


def test_build_meta_signals_can_turn_long_after_confirmation(tmp_path):
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

    first_pass = build_meta_signals(signals, region_configs, META_GROUPS, str(tmp_path))
    first_meta = first_pass["brazil_soy_meta"]

    assert first_meta["trading_action"] == "FLAT"
    assert first_meta["raw_trading_action"] == "LONG"

    meta_signals = build_meta_signals(signals, region_configs, META_GROUPS, str(tmp_path))
    meta = meta_signals["brazil_soy_meta"]

    assert meta["trading_action"] == "LONG"
    assert meta["actionability"] == "Actionable"
    assert meta["vote_score"] > 0.2


def test_apply_meta_signal_persistence_holds_pending_flip():
    raw_signal = {
        "signal": "Brazil soy meta-short",
        "confidence": "Medium",
        "actionability": "Actionable",
        "trading_action": "SHORT",
    }
    previous_state = {
        "live_action": "LONG",
        "pending_action": "FLAT",
        "pending_count": 0,
    }

    persisted, next_state = apply_meta_signal_persistence(raw_signal, previous_state, confirmations_required=2)

    assert persisted["trading_action"] == "LONG"
    assert persisted["raw_trading_action"] == "SHORT"
    assert next_state["pending_action"] == "SHORT"
    assert next_state["pending_count"] == 1


def test_update_portfolio_uses_meta_signal_not_subregions(tmp_path):
    portfolio = MultiAssetPortfolio(output_base=str(tmp_path))
    signals = {
        "brazil_soy_central": {
            "portfolio_trade": False,
            "actionability": "Actionable",
            "trading_action": "LONG",
            "instruments": ["Soybeans"],
            "signal": "Long crop",
        },
        "brazil_soy_meta": {
            "portfolio_trade": True,
            "actionability": "Actionable",
            "trading_action": "SHORT",
            "instruments": ["Soybeans"],
            "signal": "Brazil soy meta-short",
        },
    }

    actions = update_portfolio_with_signals(portfolio, signals, {"Soybeans": 1200.0})

    assert len(actions) == 1
    assert actions[0]["action"] == "OPEN_SHORT"
    assert actions[0]["region"] == "brazil_soy_meta"
