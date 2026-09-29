import streamlit as st
import numpy as np
import cv2
import math
import time
import io
import os
import tempfile
import pandas as pd
from datetime import datetime, timezone

st.set_page_config(
    page_title="Crane Hand-Signal Robotic Relay Demo",
    layout="wide",
)

# ==============================================================================
# Global constants
# ==============================================================================

CYCLE_SECONDS = 120.0
FRAME_W = 960
FRAME_H = 540
GROUND_Y = 455
RIGGER_X = 250

LIGHTING_OPTIONS = [
    "Daylight",
    "Overcast",
    "Glare",
    "Night / floodlights",
]

EMERGENCY_SIGNALS = {
    "EMERGENCY_STOP",
    "DOG_EVERYTHING",
}

MOTION_SIGNALS = {
    "HOIST",
    "LOWER",
    "SWING",
    "TROLLEY",
    "TRAVEL",
    "BOOM_UP",
    "BOOM_DOWN",
    "EXTEND_BOOM",
    "RETRACT_BOOM",
    "ROTATE",
}

# ------------------------------------------------------------------------------
# 1) ANSI/OSHA-style standard crane hand-signal reference set
#    represented as the target whole-body gesture vocabulary.
# ------------------------------------------------------------------------------

STANDARD_VOCABULARY = [
    {
        "signal": "HOIST",
        "meaning": "Raise load",
        "motion": "Fist above head, thumb up, small circular motion",
        "control": "HOIST_UP",
    },
    {
        "signal": "LOWER",
        "meaning": "Lower load",
        "motion": "Arm downward, fist, thumb down, small circular motion",
        "control": "HOIST_DOWN",
    },
    {
        "signal": "SWING",
        "meaning": "Rotate superstructure / swing load",
        "motion": "Arm extended horizontally, point direction, small circles",
        "control": "SWING",
    },
    {
        "signal": "TRAVEL",
        "meaning": "Move crane by travel function",
        "motion": "Both arms extended, palms up, repeated motion over shoulder",
        "control": "TRAVEL",
    },
    {
        "signal": "TROLLEY",
        "meaning": "Move trolley",
        "motion": "Forearm vertical, hand moves horizontally in desired direction",
        "control": "TROLLEY",
    },
    {
        "signal": "BOOM_UP",
        "meaning": "Raise boom",
        "motion": "Arm extended, thumb up, boom-up gesture",
        "control": "BOOM_UP",
    },
    {
        "signal": "BOOM_DOWN",
        "meaning": "Lower boom",
        "motion": "Arm extended, thumb down, boom-down gesture",
        "control": "BOOM_DOWN",
    },
    {
        "signal": "EXTEND_BOOM",
        "meaning": "Extend telescopic boom",
        "motion": "Hands apart, thumbs outward",
        "control": "BOOM_EXTEND",
    },
    {
        "signal": "RETRACT_BOOM",
        "meaning": "Retract telescopic boom",
        "motion": "Hands together, thumbs inward",
        "control": "BOOM_RETRACT",
    },
    {
        "signal": "ROTATE",
        "meaning": "Rotate load / controlled rotation",
        "motion": "Arm extended, index finger making circular motion",
        "control": "ROTATE",
    },
    {
        "signal": "SLOW",
        "meaning": "Slow / controlled motion",
        "motion": "One flat hand below moving hand",
        "control": "SLOW_MODE",
    },
    {
        "signal": "STOP",
        "meaning": "Stop current function",
        "motion": "Arm extended, palm down, held steady",
        "control": "STOP",
    },
    {
        "signal": "EMERGENCY_STOP",
        "meaning": "Stop all functions immediately / dog everything",
        "motion": "Both arms crossed above head or rapid emergency-stop gesture",
        "control": "ALL_STOP_PRIORITY",
    },
    {
        "signal": "AUX_TAGLINE_CUE",
        "meaning": "Site auxiliary cue: tagline guidance",
        "motion": "Both hands guiding tagline direction",
        "control": "TAGLINE_GUIDANCE",
    },
]

SIGNALS = {item["signal"]: item for item in STANDARD_VOCABULARY}

DISPLAY_NAMES = {
    "NO_SIGNAL": "NO ACTIVE SIGNAL",
    "HOIST": "HOIST",
    "LOWER": "LOWER",
    "SWING": "SWING",
    "TRAVEL": "TRAVEL",
    "TROLLEY": "TROLLEY",
    "BOOM_UP": "BOOM UP",
    "BOOM_DOWN": "BOOM DOWN",
    "EXTEND_BOOM": "EXTEND BOOM",
    "RETRACT_BOOM": "RETRACT BOOM",
    "ROTATE": "ROTATE",
    "SLOW": "SLOW",
    "STOP": "STOP",
    "EMERGENCY_STOP": "EMERGENCY STOP / DOG EVERYTHING",
    "DOG_EVERYTHING": "EMERGENCY STOP / DOG EVERYTHING",
    "AUX_TAGLINE_CUE": "TAGLINE CUE",
}

# ==============================================================================
# Utility helpers
# ==============================================================================

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def lighting_penalty(lighting: str) -> float:
    return {
        "Daylight": 0.0,
        "Overcast": 0.02,
        "Glare": 0.07,
        "Night / floodlights": 0.04,
    }.get(lighting, 0.0)


def distance_penalty(distance_m: float) -> float:
    # Small penalty from 10 m to 40 m.
    return clamp(((distance_m - 10.0) / 30.0) * 0.06, 0.0, 0.08)


def identity_confidence(distance_m: float, lighting: str, ppe_worn: bool, bystander: bool) -> float:
    if (not ppe_worn) or bystander:
        return clamp(0.16 + 0.03 * abs(math.sin(distance_m * 0.65)), 0.0, 0.30)

    base = 0.999
    pen = distance_penalty(distance_m) * 0.12 + lighting_penalty(lighting) * 0.12
    wiggle = 0.0025 * math.sin(distance_m * 1.2)
    return clamp(base - pen + wiggle, 0.0, 1.0)


def signal_confidence(signal: str, distance_m: float, lighting: str, verified: bool) -> float:
    if not verified:
        return 0.08

    if signal in EMERGENCY_SIGNALS:
        base = 0.998
    else:
        base = 0.995

    pen = distance_penalty(distance_m) * 0.25 + lighting_penalty(lighting) * 0.25
    wiggle = 0.004 * math.sin(0.7 * distance_m)
    return clamp(base - pen + wiggle, 0.0, 1.0)


def normal_latency_ms(distance_m: float, sim_t: float) -> float:
    return clamp(54.0 + distance_m * 0.6 + 4.0 * abs(math.sin(sim_t * 2.1)), 40.0, 95.0)


def emergency_latency_ms(sim_t: float) -> float:
    return clamp(18.0 + 2.0 * abs(math.sin(sim_t * 7.0)), 14.0, 24.0)


def scripted_signal(phase: float):
    if phase < 15.0:
        return "NO_SIGNAL", "script"
    if phase < 35.0:
        return "HOIST", "script"
    if phase < 55.0:
        return "SWING", "script"
    if phase < 75.0:
        return "NO_SIGNAL", "script"
    if phase < 95.0:
        return "EMERGENCY_STOP", "script"
    return "NO_SIGNAL", "script"


def crane_response_text(signal: str) -> str:
    mapping = {
        "HOIST": "Hoisting load",
        "LOWER": "Lowering load",
        "SWING": "Swinging load",
        "TRAVEL": "Travel motion enabled",
        "TROLLEY": "Trolley motion enabled",
        "BOOM_UP": "Boom raising",
        "BOOM_DOWN": "Boom lowering",
        "EXTEND_BOOM": "Boom extending",
        "RETRACT_BOOM": "Boom retracting",
        "ROTATE": "Controlled rotation",
        "SLOW": "Slow-control mode",
        "STOP": "Stop current function",
        "EMERGENCY_STOP": "ALL MOTIONS STOPPED",
        "DOG_EVERYTHING": "ALL MOTIONS STOPPED",
        "AUX_TAGLINE_CUE": "Tagline guidance engaged",
    }
    return mapping.get(signal, "HOLD")


# ==============================================================================
# Simulation state
# ==============================================================================

def create_sim(settings=None, enable_logs=True):
    sim = {
        "t": 0.0,
        "last_phase": 0.0,
        "cycle": 0,
        "paused": False,
        "speed": 1.0,

        "height": 0.2,
        "swing": 0.0,
        "trolley": 0.0,
        "travel": 0.0,

        "sway_amp": 0.35,
        "sway_angle": 0.0,
        "sway_history": [],
        "excess_sway": False,

        "signal": "NO_SIGNAL",
        "last_signal": "NO_SIGNAL",
        "source": "script",
        "confidence": 0.0,

        "identity": 0.99,
        "ppe_verified": True,
        "ppe_worn": True,
        "bystander": False,

        "relay": "IDLE",
        "crane_response": "HOLD",
        "latency_ms": 0.0,
        "priority": False,
        "last_accepted": False,

        "humanoid_x": 690.0,
        "tagline": False,
        "proximity_alert": False,
        "worker_intrusion": False,

        "distance": 25,
        "lighting": "Daylight",

        "manual_signal": None,
        "manual_until": -1.0,

        "logs": [],
        "event_flags": set(),

        "metrics": {
            "class_acc": 0.987,
            "estop_latency": 19.0,
            "sway_miss": 0.8,
            "proximity": 1.0,
            "identity": 0.992,
        },

        "enable_logs": enable_logs,
        "last_wall": time.perf_counter(),
    }

    if settings:
        sim.update(settings)

    return sim


def current_settings():
    return {
        "distance": st.session_state.get("distance", 25),
        "lighting": st.session_state.get("lighting", "Daylight"),
        "ppe_worn": st.session_state.get("ppe_worn", True),
        "bystander": st.session_state.get("bystander", False),
        "worker_intrusion": st.session_state.get("worker_intrusion", False),
        "speed": st.session_state.get("speed", 1.0),
    }


def inject_signal(signal_name: str):
    sim = st.session_state.get("sim")
    if sim is None:
        return

    sim["manual_signal"] = signal_name
    if signal_name in EMERGENCY_SIGNALS:
        sim["manual_until"] = sim["t"] + 3.0
    elif signal_name == "STOP":
        sim["manual_until"] = sim["t"] + 3.0
    else:
        sim["manual_until"] = sim["t"] + 5.0


def toggle_pause():
    sim = st.session_state.get("sim")
    if sim is not None:
        sim["paused"] = not sim.get("paused", False)


def reset_demo():
    st.session_state.sim = create_sim(current_settings(), enable_logs=True)


# ==============================================================================
# Logging / compliance ledger
# ==============================================================================

def log_event(
    sim,
    event,
    signal=None,
    source=None,
    latency=None,
    response=None,
    priority=False,
    note="",
):
    if not sim.get("enable_logs", True):
        return

    entry = {
        "wall_utc": datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
        "sim_t": round(sim["t"], 2),
        "phase_t": round(sim["t"] % CYCLE_SECONDS, 2),
        "cycle": sim.get("cycle", 0) + 1,
        "event": event,
        "signal": signal or sim.get("signal", "NO_SIGNAL"),
        "source": source or sim.get("source", "script"),
        "identity": round(sim.get("identity", 0.0), 3),
        "confidence": round(sim.get("confidence", 0.0), 3),
        "latency_ms": round(latency, 1) if latency is not None else None,
        "response": response or sim.get("crane_response", ""),
        "priority": "YES" if priority else "",
        "note": note,
    }

    sim["logs"].append(entry)
    sim["logs"] = sim["logs"][-300:]


def once(sim, key):
    if key in sim["event_flags"]:
        return False
    sim["event_flags"].add(key)
    return True


# ==============================================================================
# Scene geometry helpers
# ==============================================================================

def load_x_scene(sim):
    base = 520.0 + sim.get("trolley", 0.0) * 3.5 + sim.get("travel", 0.0) * 5.0
    swing_offset = math.sin(math.radians(sim.get("swing", 0.0))) * 135.0
    sway_offset = math.sin(math.radians(sim.get("sway_angle", 0.0))) * 28.0
    return clamp(base + swing_offset + sway_offset, 80.0, FRAME_W - 80.0)


def load_y_scene(sim):
    return clamp(390.0 - sim.get("height", 0.0) * 24.0, 130.0, 390.0)


# ==============================================================================
# Simulation step
# ==============================================================================

def step_sim(sim, dt_real):
    dt_raw = max(0.0, min(1.0, dt_real))
    sim_dt = dt_raw * float(sim.get("speed", 1.0))

    if sim_dt <= 0.0:
        return

    sim["t"] += sim_dt
    phase = sim["t"] % CYCLE_SECONDS

    # New cycle reset.
    if phase < sim["last_phase"] - 1e-3:
        sim["cycle"] += 1
        sim["event_flags"].clear()

        sim["height"] = 0.2
        sim["swing"] = 0.0
        sim["trolley"] = 0.0
        sim["travel"] = 0.0
        sim["sway_amp"] = 0.35
        sim["sway_angle"] = 0.0
        sim["sway_history"] = []
        sim["excess_sway"] = False

        sim["last_signal"] = "NO_SIGNAL"
        sim["last_accepted"] = False
        sim["relay"] = "IDLE"
        sim["crane_response"] = "HOLD"
        sim["latency_ms"] = 0.0
        sim["priority"] = False

        sim["manual_signal"] = None
        sim["manual_until"] = -1.0

    sim["last_phase"] = phase

    # Certified-rigger / hi-vis PPE marker detection.
    sim["identity"] = identity_confidence(
        float(sim.get("distance", 25)),
        sim.get("lighting", "Daylight"),
        bool(sim.get("ppe_worn", True)),
        bool(sim.get("bystander", False)),
    )
    sim["ppe_verified"] = sim["identity"] >= 0.50

    # Scripted or manually injected signal.
    scripted, script_source = scripted_signal(phase)

    if sim.get("manual_signal") and sim["t"] < sim.get("manual_until", -1.0):
        signal = sim["manual_signal"]
        source = "manual-inject"
    else:
        if sim.get("manual_signal") and sim["t"] >= sim.get("manual_until", -1.0):
            sim["manual_signal"] = None
        signal = scripted
        source = script_source

    sim["signal"] = signal
    sim["source"] = source

    # Tagline cue window inside swing phase.
    sim["tagline"] = 42.0 <= phase < 55.0

    # Classification confidence.
    if signal == "NO_SIGNAL":
        sim["confidence"] = 0.0
    else:
        sim["confidence"] = signal_confidence(
            signal,
            float(sim.get("distance", 25)),
            sim.get("lighting", "Daylight"),
            sim["ppe_verified"],
        )

    accepted = bool(sim["ppe_verified"] and sim["confidence"] >= 0.55)

    signal_changed = signal != sim["last_signal"]
    accepted_changed = accepted != sim["last_accepted"]

    latency = None

    if signal != "NO_SIGNAL" and (signal_changed or accepted_changed):
        if accepted:
            if signal in EMERGENCY_SIGNALS:
                latency = emergency_latency_ms(sim["t"])
                priority = True
                note = "ZERO_ADDED_LATENCY_PRIORITY_PATH"
            else:
                latency = normal_latency_ms(float(sim.get("distance", 25)), sim["t"])
                priority = False
                note = ""

            sim["latency_ms"] = latency
            sim["priority"] = priority
            sim["relay"] = SIGNALS.get(signal, {}).get("control", "RELAY")
            sim["crane_response"] = crane_response_text(signal)

            log_event(
                sim,
                "SIGNAL_DETECTED",
                signal=signal,
                source=source,
                latency=None,
                response="Perception classified gesture",
                priority=False,
                note="",
            )

            log_event(
                sim,
                "SIGNAL_RELAY",
                signal=signal,
                source=source,
                latency=latency,
                response=sim["relay"],
                priority=priority,
                note=note,
            )

            if priority:
                sim["metrics"]["estop_latency"] = (
                    0.70 * sim["metrics"]["estop_latency"] + 0.30 * latency
                )

        else:
            sim["latency_ms"] = 0.0
            sim["priority"] = False
            sim["relay"] = "INHIBIT"
            sim["crane_response"] = "NO_RELAY_UNVERIFIED"

            log_event(
                sim,
                "SIGNAL_IGNORED",
                signal=signal,
                source=source,
                latency=None,
                response="Inhibit relay",
                priority=False,
                note="Identity or confidence below threshold",
            )

    elif signal == "NO_SIGNAL" and signal_changed:
        sim["relay"] = "IDLE"
        sim["crane_response"] = "HOLD"
        sim["priority"] = False

    sim["last_signal"] = signal
    sim["last_accepted"] = accepted

    # Crane motion relay behavior.
    if accepted:
        if signal == "HOIST":
            sim["height"] = clamp(sim["height"] + 0.35 * sim_dt, 0.0, 10.0)
        elif signal == "LOWER":
            sim["height"] = clamp(sim["height"] - 0.35 * sim_dt, 0.0, 10.0)
        elif signal == "SWING":
            sim["swing"] = clamp(sim["swing"] + 2.1 * sim_dt, -45.0, 45.0)
        elif signal == "TROLLEY":
            sim["trolley"] = clamp(sim["trolley"] + 0.6 * sim_dt, -20.0, 20.0)
        elif signal == "TRAVEL":
            sim["travel"] = clamp(sim["travel"] + 0.45 * sim_dt, -18.0, 18.0)
        elif signal == "BOOM_UP":
            sim["height"] = clamp(sim["height"] + 0.20 * sim_dt, 0.0, 10.0)
        elif signal == "BOOM_DOWN":
            sim["height"] = clamp(sim["height"] - 0.20 * sim_dt, 0.0, 10.0)

    # Load sway model.
    if 55.0 <= phase < 75.0:
        target_amp = 4.8
    elif 75.0 <= phase < 95.0:
        target_amp = 0.7
    elif accepted and signal in MOTION_SIGNALS:
        target_amp = 1.4
    else:
        target_amp = 0.45

    sim["sway_amp"] += (target_amp - sim["sway_amp"]) * min(1.0, sim_dt * 0.9)

    sway_noise = (
        0.18 * math.sin(7.7 * sim["t"]) +
        0.07 * math.sin(17.3 * sim["t"])
    )
    sim["sway_angle"] = (
        sim["sway_amp"] * math.sin(2.0 * math.pi * 0.65 * sim["t"]) + sway_noise
    )

    sim["sway_history"].append(sim["sway_angle"])
    if len(sim["sway_history"]) > 140:
        sim["sway_history"] = sim["sway_history"][-140:]

    sim["excess_sway"] = bool(
        sim["sway_amp"] > 3.0 or abs(sim["sway_angle"]) > 3.0
    )

    # Humanoid ground-path tracking locomotion.
    load_x = load_x_scene(sim)
    target_humanoid_x = clamp(load_x + 180.0, 110.0, FRAME_W - 80.0)
    sim["humanoid_x"] += (target_humanoid_x - sim["humanoid_x"]) * min(
        1.0, sim_dt * 0.8
    )

    # Worker-proximity swing-radius alert broadcast.
    if sim.get("worker_intrusion", False):
        worker_x = clamp(load_x - 95.0, 80.0, FRAME_W - 80.0)
    else:
        worker_x = float(RIGGER_X)

    standoff_px = 145.0
    sim["proximity_alert"] = bool(
        sim.get("worker_intrusion", False) or abs(worker_x - load_x) < standoff_px
    )

    # One-time demo events.
    if 42.0 <= phase < 55.0 and once(sim, "tagline_cue"):
        log_event(
            sim,
            "TAGLINE_CUE",
            signal="AUX_TAGLINE_CUE",
            source="rigger-cue",
            latency=None,
            response="Humanoid tagline guidance engaged",
            priority=False,
            note="Manipulation behavior triggered by cue gesture",
        )

    if sim["excess_sway"] and once(sim, "sway_cross_check"):
        log_event(
            sim,
            "SWAY_CROSS_CHECK",
            signal="NO_SIGNAL",
            source="onboard-estimator",
            latency=None,
            response="Excess sway flag raised",
            priority=False,
            note="Cross-checking against rigger stop intent",
        )

    if sim["proximity_alert"] and once(sim, "proximity_alert"):
        log_event(
            sim,
            "SWING_RADIUS_ALERT",
            signal="NO_SIGNAL",
            source="proximity-monitor",
            latency=None,
            response="Broadcast worker standoff warning",
            priority=True,
            note="Hard-coded load-swept-path standoff enforced",
        )

    if not sim["proximity_alert"]:
        sim["event_flags"].discard("proximity_alert")

    # KPI metrics.
    if signal != "NO_SIGNAL":
        target_acc = clamp(
            0.997
            - distance_penalty(float(sim.get("distance", 25))) * 0.35
            - lighting_penalty(sim.get("lighting", "Daylight")) * 0.25,
            0.952,
            0.999,
        )
        sim["metrics"]["class_acc"] = (
            0.90 * sim["metrics"]["class_acc"] + 0.10 * target_acc
        )

    sim["metrics"]["identity"] = (
        0.90 * sim["metrics"]["identity"] + 0.10 * sim["identity"]
    )

    sim["metrics"]["sway_miss"] = clamp(
        0.75 + 0.20 * abs(math.sin(sim["t"] * 0.23)),
        0.40,
        1.40,
    )

    sim["metrics"]["proximity"] = 1.0


# ==============================================================================
# OpenCV drawing helpers
# ==============================================================================

def put_text(
    img,
    text,
    x,
    y,
    scale=0.48,
    color=(255, 255, 255),
    thickness=1,
    shadow=True,
):
    x = int(x)
    y = int(y)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if shadow:
        cv2.putText(
            img,
            text,
            (x + 1, y + 1),
            font,
            scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_person(
    img,
    x,
    ground_y,
    signal,
    hi_vis=True,
    certified=True,
    scale=1.0,
    label=None,
):
    x = int(x)
    ground_y = int(ground_y)
    s = float(scale)

    head_r = max(4, int(8 * s))
    head_y = int(ground_y - 84 * s)
    neck_y = int(ground_y - 70 * s)
    shoulder_y = int(ground_y - 65 * s)
    hip_y = int(ground_y - 32 * s)

    l_sh = (int(x - 13 * s), shoulder_y)
    r_sh = (int(x + 13 * s), shoulder_y)
    l_hip = (int(x - 8 * s), hip_y)
    r_hip = (int(x + 8 * s), hip_y)
    l_foot = (int(x - 12 * s), ground_y)
    r_foot = (int(x + 12 * s), ground_y)

    l_elb = (int(x - 18 * s), int(ground_y - 50 * s))
    l_hand = (int(x - 21 * s), int(ground_y - 30 * s))
    r_elb = (int(x + 18 * s), int(ground_y - 50 * s))
    r_hand = (int(x + 21 * s), int(ground_y - 30 * s))

    sig = signal

    if sig == "HOIST":
        r_elb = (int(x + 15 * s), int(ground_y - 72 * s))
        r_hand = (int(x + 18 * s), int(ground_y - 95 * s))
    elif sig == "LOWER":
        r_elb = (int(x + 18 * s), int(ground_y - 42 * s))
        r_hand = (int(x + 22 * s), int(ground_y - 12 * s))
    elif sig == "SWING":
        r_elb = (int(x + 28 * s), int(ground_y - 60 * s))
        r_hand = (int(x + 52 * s), int(ground_y - 60 * s))
    elif sig == "STOP":
        l_elb = (int(x - 25 * s), shoulder_y)
        r_elb = (int(x + 25 * s), shoulder_y)
        l_hand = (int(x - 38 * s), shoulder_y)
        r_hand = (int(x + 38 * s), shoulder_y)
    elif sig in EMERGENCY_SIGNALS:
        l_elb = (int(x - 18 * s), int(ground_y - 75 * s))
        r_elb = (int(x + 18 * s), int(ground_y - 75 * s))
        l_hand = (int(x + 6 * s), int(ground_y - 92 * s))
        r_hand = (int(x - 6 * s), int(ground_y - 92 * s))
    elif sig == "AUX_TAGLINE_CUE":
        l_elb = (int(x - 5 * s), int(ground_y - 55 * s))
        r_elb = (int(x + 25 * s), int(ground_y - 55 * s))
        l_hand = (int(x + 32 * s), int(ground_y - 48 * s))
        r_hand = (int(x + 52 * s), int(ground_y - 48 * s))

    # Hi-vis PPE vest.
    vest_color = (0, 230, 90) if hi_vis else (130, 130, 130)
    cv2.rectangle(
        img,
        (int(x - 14 * s), shoulder_y - 4),
        (int(x + 14 * s), hip_y),
        vest_color,
        -1,
    )

    # Skeleton overlay.
    skeleton_color = (255, 255, 0)
    lw = 2

    cv2.line(img, (x, neck_y), (x, hip_y), skeleton_color, lw)
    cv2.line(img, l_sh, r_sh, skeleton_color, lw)

    cv2.line(img, l_sh, l_hip, skeleton_color, lw)
    cv2.line(img, r_sh, r_hip, skeleton_color, lw)

    cv2.line(img, l_hip, l_foot, skeleton_color, lw)
    cv2.line(img, r_hip, r_foot, skeleton_color, lw)

    cv2.line(img, l_sh, l_elb, skeleton_color, lw)
    cv2.line(img, l_elb, l_hand, skeleton_color, lw)

    cv2.line(img, r_sh, r_elb, skeleton_color, lw)
    cv2.line(img, r_elb, r_hand, skeleton_color, lw)

    # Head / helmet.
    cv2.circle(img, (x, head_y), head_r, (0, 210, 250), -1)

    # Signal motion indicators.
    if sig == "HOIST":
        cv2.circle(img, r_hand, int(10 * s), (0, 255, 255), 2)
    elif sig == "SWING":
        cv2.ellipse(
            img,
            (int(x + 55 * s), shoulder_y),
            (int(18 * s), int(8 * s)),
            0,
            -40,
            220,
            (0, 255, 255),
            2,
        )
    elif sig in EMERGENCY_SIGNALS:
        cv2.rectangle(
            img,
            (int(x - 35 * s), int(ground_y - 100 * s)),
            (int(x + 35 * s), ground_y),
            (0, 0, 255),
            2,
        )

    # Identity / PPE marker box.
    box_color = (0, 220, 0) if certified else (0, 0, 255)
    x1 = int(x - 34 * s)
    y1 = max(6, head_y - head_r - 20)
    x2 = int(x + 34 * s)
    y2 = ground_y + 2

    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

    if label:
        put_text(
            img,
            label,
            x1,
            max(14, y1 - 6),
            0.40,
            box_color,
            1,
        )


def draw_humanoid(
    img,
    x,
    ground_y,
    tagline=False,
    load_x=None,
    load_y=None,
):
    x = int(x)
    ground_y = int(ground_y)

    # Body.
    cv2.rectangle(
        img,
        (x - 18, ground_y - 60),
        (x + 18, ground_y - 12),
        (200, 170, 70),
        -1,
    )

    # Head / sensor mast.
    cv2.rectangle(
        img,
        (x - 12, ground_y - 78),
        (x + 12, ground_y - 60),
        (220, 220, 220),
        -1,
    )

    # Lidar / camera.
    cv2.circle(img, (x, ground_y - 84), 7, (0, 255, 255), -1)

    # Base / tracks.
    cv2.rectangle(
        img,
        (x - 26, ground_y - 12),
        (x + 26, ground_y),
        (80, 80, 80),
        -1,
    )

    put_text(
        img,
        "HUMANOID VANTAGE",
        x - 58,
        ground_y - 98,
        0.38,
        (255, 200, 80),
        1,
    )

    if tagline and load_x is not None and load_y is not None:
        cv2.line(
            img,
            (x, ground_y - 45),
            (int(load_x), int(load_y) + 55),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        put_text(
            img,
            "TAGLINE GUIDANCE",
            x - 45,
            ground_y - 115,
            0.38,
            (0, 255, 255),
            1,
        )


def draw_status_panel(img, sim):
    panel_x = 12
    panel_y = 12
    panel_w = 420
    panel_h = 182

    cv2.rectangle(
        img,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (25, 25, 25),
        -1,
    )
    cv2.rectangle(
        img,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (200, 200, 200),
        1,
    )

    phase = sim["t"] % CYCLE_SECONDS
    y = panel_y + 24

    put_text(
        img,
        f"T+{phase:05.1f}s  Cycle {sim.get('cycle', 0) + 1}  Speed {sim.get('speed', 1.0):.1f}x",
        panel_x + 10,
        y,
        0.48,
        (255, 255, 255),
        1,
    )
    y += 22

    sig_disp = DISPLAY_NAMES.get(sim.get("signal", "NO_SIGNAL"), sim.get("signal", "NO_SIGNAL"))
    signal_color = (80, 255, 120)
    if sim.get("signal") in EMERGENCY_SIGNALS:
        signal_color = (60, 60, 255)
    elif sim.get("signal") == "NO_SIGNAL":
        signal_color = (200, 200, 200)

    put_text(
        img,
        f"Signal: {sig_disp}  [{sim.get('source', 'script')}]",
        panel_x + 10,
        y,
        0.48,
        signal_color,
        1,
    )
    y += 22

    verified_txt = "VERIFIED" if sim.get("ppe_verified", False) else "NOT VERIFIED"
    verified_color = (80, 255, 120) if sim.get("ppe_verified", False) else (60, 60, 255)

    put_text(
        img,
        f"Conf {sim.get('confidence', 0.0):.2f} | ID {sim.get('identity', 0.0):.2f} | {verified_txt}",
        panel_x + 10,
        y,
        0.46,
        verified_color,
        1,
    )
    y += 22

    put_text(
        img,
        f"Relay: {sim.get('relay', 'IDLE')} | Latency: {sim.get('latency_ms', 0.0):.0f} ms | Priority: {'YES' if sim.get('priority') else 'NO'}",
        panel_x + 10,
        y,
        0.44,
        (220, 220, 220),
        1,
    )
    y += 22

    put_text(
        img,
        f"Crane response: {sim.get('crane_response', 'HOLD')}",
        panel_x + 10,
        y,
        0.46,
        (170, 220, 255),
        1,
    )
    y += 22

    sway_color = (60, 60, 255) if sim.get("excess_sway", False) else (120, 255, 120)
    put_text(
        img,
        f"Sway: {sim.get('sway_angle', 0.0):+04.1f} deg | Amp: {sim.get('sway_amp', 0.0):.1f} deg | Excess: {'YES' if sim.get('excess_sway') else 'NO'}",
        panel_x + 10,
        y,
        0.42,
        sway_color,
        1,
    )

    if sim.get("priority", False):
        put_text(
            img,
            "ZERO-ADDED-LATENCY PRIORITY PATH ACTIVE",
            panel_x + 10,
            panel_y + panel_h - 12,
            0.44,
            (50, 50, 255),
            1,
        )


def draw_sway_panel(img, sim):
    x0 = 12
    y0 = FRAME_H - 112
    w = 340
    h = 98

    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (180, 180, 180), 1)

    put_text(
        img,
        "Onboard load-sway estimator",
        x0 + 10,
        y0 + 18,
        0.42,
        (220, 220, 220),
        1,
    )

    hist = sim.get("sway_history", [])
    if len(hist) > 1:
        pts = []
        mid_y = y0 + 58
        amp_px = 24.0

        for i, val in enumerate(hist[-110:]):
            px = x0 + 12 + int((w - 24) * i / max(1, len(hist[-110:]) - 1))
            py = int(mid_y - clamp(val / 6.0, -1.0, 1.0) * amp_px)
            pts.append((px, py))

        color = (60, 60, 255) if sim.get("excess_sway", False) else (80, 255, 120)
        cv2.polylines(
            img,
            [np.array(pts, dtype=np.int32)],
            False,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.line(
        img,
        (x0 + 10, y0 + 58),
        (x0 + w - 10, y0 + 58),
        (120, 120, 120),
        1,
    )


def add_zoom_inset(img, rigger_x, ground_y):
    roi_w = 250
    roi_h = 175

    x1 = max(0, int(rigger_x - 115))
    y1 = max(0, int(ground_y - roi_h))
    x2 = min(FRAME_W, x1 + roi_w)
    y2 = min(FRAME_H, y1 + roi_h)

    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return

    roi = cv2.resize(roi, (240, 150), interpolation=cv2.INTER_LINEAR)

    px = FRAME_W - 262
    py = 18

    img[py:py + 150, px:px + 240] = roi
    cv2.rectangle(img, (px, py), (px + 240, py + 150), (0, 255, 255), 2)

    put_text(
        img,
        "Zoom camera pair | calibrated 10-40 m",
        px,
        py + 164,
        0.40,
        (0, 255, 255),
        1,
    )


# ==============================================================================
# Full frame renderer
# ==============================================================================

def render_frame(sim):
    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    lighting = sim.get("lighting", "Daylight")

    if lighting == "Night / floodlights":
        sky = (60, 45, 25)
        ground = (60, 60, 65)
        crane_color = (190, 190, 200)
        cable_color = (200, 200, 210)
    elif lighting == "Glare":
        sky = (185, 225, 255)
        ground = (95, 115, 105)
        crane_color = (130, 130, 140)
        cable_color = (180, 180, 190)
    elif lighting == "Overcast":
        sky = (210, 210, 220)
        ground = (105, 110, 105)
        crane_color = (150, 150, 160)
        cable_color = (190, 190, 200)
    else:
        sky = (250, 195, 110)
        ground = (85, 120, 160)
        crane_color = (170, 170, 180)
        cable_color = (230, 230, 235)

    img[:GROUND_Y] = sky
    img[GROUND_Y:] = ground

    cv2.line(img, (0, GROUND_Y), (FRAME_W, GROUND_Y), (210, 210, 210), 2)

    # Crane mast and boom.
    cv2.rectangle(img, (138, 120), (162, GROUND_Y), crane_color, -1)
    cv2.line(img, (150, 120), (830, 120), crane_color, 9)
    cv2.line(img, (150, 120), (80, 190), crane_color, 7)
    cv2.rectangle(img, (55, 185), (105, 215), (70, 70, 90), -1)

    trolley_x = int(
        520.0 + sim.get("trolley", 0.0) * 3.5 + sim.get("travel", 0.0) * 5.0
    )
    load_x = load_x_scene(sim)
    hook_y = load_y_scene(sim)

    # Trolley.
    cv2.rectangle(img, (trolley_x - 14, 116), (trolley_x + 14, 132), (80, 80, 90), -1)

    # Cable.
    cv2.line(
        img,
        (trolley_x, 132),
        (int(load_x), int(hook_y)),
        cable_color,
        2,
        cv2.LINE_AA,
    )

    # Load.
    cv2.rectangle(
        img,
        (int(load_x - 48), int(hook_y + 10)),
        (int(load_x + 48), int(hook_y + 62)),
        (60, 60, 205),
        -1,
    )
    cv2.rectangle(
        img,
        (int(load_x - 48), int(hook_y + 10)),
        (int(load_x + 48), int(hook_y + 62)),
        (20, 20, 20),
        2,
    )
    put_text(
        img,
        "TEST LOAD",
        int(load_x - 34),
        int(hook_y + 42),
        0.42,
        (255, 255, 255),
        1,
    )

    # Load-swept-path standoff.
    standoff_color = (0, 0, 255) if sim.get("proximity_alert", False) else (0, 220, 255)
    cv2.ellipse(
        img,
        (int(load_x), GROUND_Y - 4),
        (155, 36),
        0,
        0,
        360,
        standoff_color,
        3 if sim.get("proximity_alert", False) else 2,
    )
    put_text(
        img,
        "LOAD-SWEPT PATH STANDOFF",
        int(load_x - 115),
        GROUND_Y - 55,
        0.38,
        standoff_color,
        1,
    )

    # Optional worker intrusion.
    if sim.get("worker_intrusion", False):
        worker_x = clamp(load_x - 95.0, 80.0, FRAME_W - 80.0)
        draw_person(
            img,
            worker_x,
            GROUND_Y,
            "STOP",
            hi_vis=False,
            certified=False,
            scale=0.92,
            label="WORKER IN RADIUS",
        )

    # Rigger / bystander.
    phase = sim["t"] % CYCLE_SECONDS
    rigger_draw_signal = sim.get("signal", "NO_SIGNAL")

    if (
        sim.get("tagline", False)
        and 42.0 <= phase < 52.0
        and rigger_draw_signal not in EMERGENCY_SIGNALS
    ):
        rigger_draw_signal = "AUX_TAGLINE_CUE"

    hi_vis = bool(sim.get("ppe_worn", True)) and not bool(sim.get("bystander", False))
    rigger_label = (
        "CERTIFIED RIGGER"
        if sim.get("ppe_verified", False)
        else "BYSTANDER / NO RELAY"
    )

    draw_person(
        img,
        RIGGER_X,
        GROUND_Y,
        rigger_draw_signal,
        hi_vis=hi_vis,
        certified=bool(sim.get("ppe_verified", False)),
        scale=1.0,
        label=rigger_label,
    )

    # Active signal under rigger.
    sig_disp = DISPLAY_NAMES.get(sim.get("signal", "NO_SIGNAL"), sim.get("signal", "NO_SIGNAL"))
    put_text(
        img,
        sig_disp,
        RIGGER_X - 55,
        GROUND_Y + 22,
        0.44,
        (0, 255, 255),
        1,
    )

    # Humanoid.
    draw_humanoid(
        img,
        sim.get("humanoid_x", 700.0),
        GROUND_Y,
        tagline=bool(sim.get("tagline", False)),
        load_x=load_x,
        load_y=hook_y,
    )

    # Alerts.
    if sim.get("proximity_alert", False):
        put_text(
            img,
            "SWING-RADIUS ALERT BROADCAST",
            450,
            220,
            0.50,
            (50, 50, 255),
            2,
        )

    if sim.get("excess_sway", False):
        put_text(
            img,
            "LOAD-SWAY CROSS-CHECK FLAG",
            450,
            245,
            0.50,
            (50, 160, 255),
            2,
        )

    # Status panel and zoom inset.
    draw_status_panel(img, sim)
    add_zoom_inset(img, RIGGER_X, GROUND_Y)

    # Sway panel after zoom so zoom does not copy it.
    draw_sway_panel(img, sim)

    # Camera calibration caption.
    put_text(
        img,
        f"Wide-FOV + zoom pair | Distance: {sim.get('distance', 25):.0f} m | Lighting: {sim.get('lighting', 'Daylight')}",
        360,
        FRAME_H - 12,
        0.42,
        (220, 220, 220),
        1,
    )

    return img


# ==============================================================================
# Video generation / live demo recorder
# ==============================================================================

def generate_demo_video(duration_s=60, fps=8, settings=None):
    duration_s = int(clamp(duration_s, 1, 60))
    fps = int(clamp(fps, 4, 15))

    # Fit the full 120-second scripted scenario into the requested clip length.
    video_speed = clamp(CYCLE_SECONDS / float(duration_s), 1.0, 4.0)

    sim = create_sim(settings=settings, enable_logs=False)
    sim["speed"] = video_speed

    dt = 1.0 / float(fps)
    out_size = (640, 360)
    tmp_path = None

    try:
        import imageio

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()

        writer = imageio.get_writer(
            tmp_path,
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            pixelformat="yuv420p",
        )

        total_frames = int(duration_s * fps)

        for _ in range(total_frames):
            step_sim(sim, dt)
            frame = render_frame(sim)
            frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb)

        writer.close()

        with open(tmp_path, "rb") as f:
            data = f.read()

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        return data, "video/mp4", "mp4"

    except Exception as primary_error:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # Fallback to GIF if MP4 encoding is unavailable.
        try:
            from PIL import Image

            sim = create_sim(settings=settings, enable_logs=False)
            sim["speed"] = video_speed

            frames = []
            capped_duration = min(duration_s, 12)
            total_frames = int(capped_duration * fps)

            for _ in range(total_frames):
                step_sim(sim, dt)
                frame = render_frame(sim)
                frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))

            buf = io.BytesIO()
            frames[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )

            return buf.getvalue(), "image/gif", "gif"

        except Exception:
            st.error(f"Video generation failed: {primary_error}")
            return None, None, None


# ==============================================================================
# Streamlit callbacks and sidebar
# ==============================================================================

st.sidebar.header("Site / camera controls")

st.sidebar.slider(
    "Rigger-to-camera distance (m)",
    min_value=10,
    max_value=40,
    value=25,
    step=1,
    key="distance",
)

st.sidebar.selectbox(
    "Outdoor lighting",
    LIGHTING_OPTIONS,
    index=0,
    key="lighting",
)

st.sidebar.checkbox(
    "Certified rigger wearing hi-vis PPE marker",
    value=True,
    key="ppe_worn",
)

st.sidebar.checkbox(
    "Simulate incidental bystander instead of rigger",
    value=False,
    key="bystander",
)

st.sidebar.checkbox(
    "Simulate worker inside swing radius",
    value=False,
    key="worker_intrusion",
)

st.sidebar.radio(
    "Simulation speed",
    options=[0.5, 1.0, 2.0, 4.0],
    index=1,
    horizontal=True,
    key="speed",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Manual signal injection")

man_cols = st.sidebar.columns(2)

man_cols[0].button(
    "HOIST",
    use_container_width=True,
    on_click=inject_signal,
    args=("HOIST",),
)

man_cols[1].button(
    "LOWER",
    use_container_width=True,
    on_click=inject_signal,
    args=("LOWER",),
)

man_cols[0].button(
    "SWING",
    use_container_width=True,
    on_click=inject_signal,
    args=("SWING",),
)

man_cols[1].button(
    "STOP",
    use_container_width=True,
    on_click=inject_signal,
    args=("STOP",),
)

st.sidebar.button(
    "EMERGENCY STOP / DOG EVERYTHING",
    type="primary",
    use_container_width=True,
    on_click=inject_signal,
    args=("EMERGENCY_STOP",),
)

st.sidebar.markdown("---")

side_cols = st.sidebar.columns(2)

side_cols[0].button(
    "Pause / Resume",
    use_container_width=True,
    on_click=toggle_pause,
)

side_cols[1].button(
    "Reset demo",
    use_container_width=True,
    on_click=reset_demo,
)

# Initialize simulation after sidebar widgets exist.
if "sim" not in st.session_state:
    st.session_state.sim = create_sim(current_settings(), enable_logs=True)

if "video_data" not in st.session_state:
    st.session_state.video_data = None

if "video_mime" not in st.session_state:
    st.session_state.video_mime = None

if "video_ext" not in st.session_state:
    st.session_state.video_ext = "mp4"


# ==============================================================================
# Main page header
# ==============================================================================

st.title("ANSI/OSHA Crane Hand-Signal → Humanoid Relay Demo")

st.caption(
    "Self-contained synthetic live demo. It auto-starts the 2-minute scripted rigging scenario, "
    "simulates camera perception, rigger identity verification, crane relay logic, load-sway cross-check, "
    "humanoid vantage tracking, proximity alerts, and compliance logging."
)

st.warning(
    "Simulation only. This demo does not control a real crane. Any real crane-control integration "
    "requires certified functional safety, site approval, and independent verification."
)


# ==============================================================================
# Live demo fragment
# ==============================================================================

def live_demo_inner():
    sim = st.session_state.sim

    # Keep simulation synchronized with sidebar controls.
    sim["distance"] = st.session_state.get("distance", 25)
    sim["lighting"] = st.session_state.get("lighting", "Daylight")
    sim["ppe_worn"] = st.session_state.get("ppe_worn", True)
    sim["bystander"] = st.session_state.get("bystander", False)
    sim["worker_intrusion"] = st.session_state.get("worker_intrusion", False)
    sim["speed"] = st.session_state.get("speed", 1.0)

    now = time.perf_counter()
    dt = now - sim.get("last_wall", now)
    sim["last_wall"] = now

    if not sim.get("paused", False):
        step_sim(sim, dt)

    metrics = sim.get("metrics", {})

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Signal classification @ 10-40 m",
        f"{metrics.get('class_acc', 0.0) * 100.0:.1f}%",
        delta="target > 95%",
        delta_color="normal",
    )

    k2.metric(
        "Emergency-stop relay latency",
        f"{metrics.get('estop_latency', 0.0):.0f} ms",
        delta="target < 100 ms",
        delta_color="normal",
    )

    k3.metric(
        "Sway cross-check false-negative",
        f"{metrics.get('sway_miss', 0.0):.2f}%",
        delta="target < 2%",
        delta_color="normal",
    )

    k4.metric(
        "Worker-proximity alert reliability",
        f"{metrics.get('proximity', 1.0) * 100.0:.0f}%",
        delta="target 100%",
        delta_color="normal",
    )

    k5.metric(
        "Certified-rigger ID accuracy",
        f"{metrics.get('identity', 0.0) * 100.0:.1f}%",
        delta="target > 98%",
        delta_color="normal",
    )

    frame = render_frame(sim)
    st.image(frame, channels="BGR", use_container_width=True)

    with st.expander("Compliance ledger (live signal / relay / crane-response events)", expanded=True):
        if sim.get("logs"):
            df = pd.DataFrame(sim["logs"])
            df = df.iloc[::-1]
            st.dataframe(df, use_container_width=True, height=240)
        else:
            st.caption("No events yet.")


if hasattr(st, "fragment"):
    live_demo = st.fragment(run_every="450ms")(live_demo_inner)
    live_demo()
else:
    # Fallback for older Streamlit versions.
    st.components.v1.html(
        "<script>setTimeout(function(){window.parent.location.reload();}, 800);</script>",
        height=0,
    )
    live_demo_inner()


# ==============================================================================
# ANSI/OSHA vocabulary reference
# ==============================================================================

with st.expander("ANSI/OSHA target whole-body gesture vocabulary", expanded=False):
    vocab_df = pd.DataFrame(STANDARD_VOCABULARY)
    st.dataframe(vocab_df, use_container_width=True, height=320)


# ==============================================================================
# Live demo video recorder / download
# ==============================================================================

st.subheader("Live demo video recorder")

st.caption(
    "The live panel above is already running. Use this recorder to generate and download "
    "a real MP4 video of the demo. For a 60-second clip, the recorder renders the full "
    "120-second scripted scenario at 2x playback so the hoist, swing, sway cross-check, "
    "and emergency-stop events are included."
)

rec_col1, rec_col2 = st.columns([1, 2])

with rec_col1:
    record_duration = st.slider(
        "Recording duration (seconds)",
        min_value=5,
        max_value=60,
        value=60,
        step=5,
        key="record_duration",
    )

    record_fps = st.select_slider(
        "Recording FPS",
        options=[6, 8, 10],
        value=8,
        key="record_fps",
    )

    generate_clicked = st.button(
        "Generate downloadable live-demo recording",
        type="primary",
        use_container_width=True,
    )

with rec_col2:
    st.markdown(
        """
        **Recorder notes**
        - The demo auto-starts immediately; no blank dashboard.
        - The emergency-stop path is prioritized and logged.
        - The compliance ledger records every signal, relay, and crane-response event.
        - The video renderer uses the current sidebar site/camera settings.
        """
    )

if generate_clicked:
    settings = current_settings()

    with st.spinner(f"Rendering {record_duration}s live demo video..."):
        data, mime, ext = generate_demo_video(
            duration_s=record_duration,
            fps=record_fps,
            settings=settings,
        )

    if data is not None:
        st.session_state.video_data = data
        st.session_state.video_mime = mime
        st.session_state.video_ext = ext

if st.session_state.get("video_data") is not None:
    mime = st.session_state.get("video_mime", "video/mp4")
    ext = st.session_state.get("video_ext", "mp4")

    st.success("Demo video rendered. Preview and download below.")

    if mime == "video/mp4":
        st.video(st.session_state.video_data)
    else:
        st.image(st.session_state.video_data, caption="Generated demo recording")

    st.download_button(
        label=f"Download live demo recording (.{ext})",
        data=st.session_state.video_data,
        file_name=f"crane_hand_signal_demo.{ext}",
        mime=mime,
        type="primary",
        use_container_width=True,
    )