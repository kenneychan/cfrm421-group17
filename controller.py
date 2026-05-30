"""
CONTROLLER -- runs each model through the three stages in order:
    1. train()        -- fit on training set
    2. refine_model() -- tune on refinement set
    3. predict()      -- signal from evaluation set

Then aggregates all signals via majority vote.
"""

from model import BaseModel


def run_ensemble(models: list[BaseModel]) -> dict:
    """
    Run all models through train -> refine_model -> predict and
    aggregate via majority vote.

    Returns:
        {
            "signals":         { model_name: 1 or -1 },
            "votes_up":        int,
            "votes_down":      int,
            "ensemble_signal": 1, -1, or 0 (tie),
            "consensus":       "UP" | "DOWN" | "TIE"
        }
    """
    signals = {}
    for model in models:
        try:
            model.train()
            model.refine_model()
            signal = model.predict()
            assert signal in (1, -1), f"{model.name} returned {signal}, expected 1 or -1"
            signals[model.name] = signal
        except Exception as e:
            print(f"[WARN] {model.name} failed: {e}")
            signals[model.name] = None

    valid      = [s for s in signals.values() if s is not None]
    votes_up   = valid.count(1)
    votes_down = valid.count(-1)

    if votes_up > votes_down:
        ensemble_signal, consensus = 1, "UP"
    elif votes_down > votes_up:
        ensemble_signal, consensus = -1, "DOWN"
    else:
        ensemble_signal, consensus = 0, "TIE"

    return {
        "signals":         signals,
        "votes_up":        votes_up,
        "votes_down":      votes_down,
        "ensemble_signal": ensemble_signal,
        "consensus":       consensus,
    }
