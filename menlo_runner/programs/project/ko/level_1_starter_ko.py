from __future__ import annotations

"""Level 1 프로젝트 스타터입니다.

이 파일은 완성된 해답이 아니라 최소 scaffold입니다.

SUPPORT CODE 영역은 반복해서 작성할 필요가 없는 wrapper, 자료 구조,
schema validation을 제공합니다. STUDENT TODO 영역은 팀이 직접 설계하고,
개선하고, 테스트하고, 발표에서 설명해야 하는 부분입니다.

Level 1 규칙: `scene_state`와 정확한 entity ID는 사용할 수 없습니다.
coordinate `go_to`는 학생 시스템이 직접 추정하거나 기록한 좌표에만
사용할 수 있습니다.
"""

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from menlo_runner.basics import screenshot
from menlo_runner.completion import CompletionConfig, CompletionTracker
from menlo_runner.config import load_config
from menlo_runner.llm import ask_vlm, call_llm
from menlo_runner.perception import decode_jpeg, detect_color_blobs
from menlo_runner.scene import delivered_cube_ids, held_cube_info


# ------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 고정 환경 설정 및 규칙
# ---------------------------------------------------------------------------
TASK = "Find and sort cubes from the source area into their matching destination pads."

DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}
SIGN_TO_PAD_COLOR = {sign: color for color, sign in DESTINATION_SIGN_RULES.items()}
LANDMARK_LETTERS = {"A", "B", "C", "D", "E"}
SIGNAGE_NOTE = (
    "A는 conveyor/cube source area이며 destination이 아닙니다. "
    "Destination sign은 B red, C green, D blue, E yellow입니다."
)
SIGN_SCAN_YAWS = (-0.9, 0.0, 0.9)
TARGET_SCAN_YAWS = (-0.85, 0.85, 0.0)
COLOR_SCAN_YAWS = (-0.9, -0.45, 0.0, 0.45, 0.9)
CUBE_DISTANCE_K = 70.0
SIGN_HFOV_HALF_DEG = 30.0
CAMERA_HFOV_DEG = 60.0
DEFAULT_CAMERA_HEIGHT_M = 1.25
CAMERA_PITCH_OFFSET_RAD = 0.0
GROUND_DISTANCE_MIN_M = 0.45
GROUND_DISTANCE_MAX_M = 7.0
NEAR_SOURCE_M = 1.3
NEAR_PAD_M = 1.4
CAPTURE_DIR = Path("outputs/level1_captures")
SOURCE_CUBE_AREA_THRESHOLD = 1200
SOURCE_CUBE_COUNT_THRESHOLD = 3
MIN_SIGN_BASELINE_M = 0.65
MIN_SIGN_TRIANGULATION_ANGLE_DEG = 8.0
MAX_SIGN_RAY_DISTANCE_M = 14.0
MAX_SIGN_ESTIMATE_SPREAD_M = 1.8
SIGN_OBSERVATION_LIMIT = 16
TRIANGULATION_FORWARD_M = 0.15
TRIANGULATION_SIDE_M = 1.05

ALLOWED_NEXT_ACTIONS = {
    "search_cube",
    "navigate_to_cube",
    "pick_cube",
    "search_pad",
    "navigate_to_pad",
    "place_cube",
    "recover",
    "skip_target",
    "stop",
}

@dataclass
class AgentDecision:
    next_action: str
    target_color: str | None = None
    reason: str = ""
    recovery_strategy: str | None = None

@dataclass
class AgentMemory:
    delivered_count: int = 0
    held_color: str | None = None
    active_color: str | None = None
    stage: str = "need_cube"
    search_turns: int = 0
    failed_attempts: dict[str, int] = field(default_factory=dict)
    completed_colors: list[str] = field(default_factory=list)
    skipped_colors: list[str] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)
    
    # [알고리즘 핵심] 한번 탐색 및 발견된 객체/패드의 World 좌표를 기록하는 저장소
    known_locations: dict[str, tuple[float, float]] = field(default_factory=dict)
    location_confidence: dict[str, float] = field(default_factory=dict)
    sign_observations: dict[str, list[Any]] = field(default_factory=dict)
    rejected_locations: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    last_vlm_summary: str = ""
    last_action_at: float = 0.0
    capture_index: int = 0
    tokamak_api_key: str = ""

@dataclass
class Observation:
    robot_status: Any
    detections: list[Any]
    signs: list[Any] = field(default_factory=list)
    note: str = ""
    vlm_summary: str = ""

@dataclass(frozen=True)
class ScannedDetection:
    color: str
    angle_deg: float
    blob_area: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    frame_size: tuple[int, int]
    head_yaw: float
    head_pitch: float

    @property
    def full_bearing_deg(self) -> float:
        return self.angle_deg + math.degrees(self.head_yaw)

@dataclass(frozen=True)
class SignSighting:
    letter: str
    bearing_deg: float
    confidence: float
    head_yaw: float
    position_hint: str = ""
    raw: str = ""

@dataclass(frozen=True)
class SignObservation:
    letter: str
    robot_xy: tuple[float, float]
    absolute_bearing_rad: float
    confidence: float

# ---------------------------------------------------------------------------
# 기본 제공 및 유틸리티 파서
# ---------------------------------------------------------------------------
def parse_agent_decision(text: str) -> AgentDecision | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    next_action = data.get("next_action")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        return None

    return AgentDecision(
        next_action=next_action,
        target_color=data.get("target_color"),
        reason=str(data.get("reason", "")),
        recovery_strategy=data.get("recovery_strategy"),
    )

def build_decision_context(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    visible = [
        {
            "color": detection.color,
            "angle_deg": detection.angle_deg,
            "full_bearing_deg": round(detection.full_bearing_deg, 1),
            "blob_area": detection.blob_area,
            "bbox": detection.bbox,
        }
        for detection in observation.detections
    ]
    return {
        "task": task,
        "visible_targets": visible,
        "visible_signs": [
            {
                "letter": sign.letter,
                "bearing_deg": round(sign.bearing_deg, 1),
                "confidence": round(sign.confidence, 2),
                "position_hint": sign.position_hint,
            }
            for sign in observation.signs
        ],
        "held_color": memory.held_color,
        "active_color": memory.active_color,
        "stage": memory.stage,
        "delivered_count": memory.delivered_count,
        "known_locations": {k: [round(v[0], 2), round(v[1], 2)] for k, v in memory.known_locations.items()},
        "sign_observation_counts": {k: len(v) for k, v in sorted(memory.sign_observations.items())},
        "last_result": last_result,
        "note": observation.note,
        "signage_note": SIGNAGE_NOTE,
    }

# ---------------------------------------------------------------------------
# 로봇 제어 Low-Level SDK Wrappers
# ---------------------------------------------------------------------------
async def get_robot_status(ctx: Any) -> Any:
    return await ctx.state("robot_status")

async def get_camera_frame(ctx: Any) -> bytes:
    return await ctx.get_vision("pov")

def build_signage_vlm_prompt(needed_letter: str | None = None) -> str:
    target = f" The robot is looking for sign {needed_letter}." if needed_letter else ""
    return (
        "Read the warehouse spot signs visible in this robot POV image. "
        "The signs are large labeled spots A, B, C, D, and E. "
        f"{SIGNAGE_NOTE} "
        "Return ONLY JSON like "
        '{"signs":[{"letter":"A","position":"left|center|right|far_left|far_right",'
        '"x_position":0.0,"confidence":0.0}]}. '
        "x_position is the horizontal center of the sign in the image from 0.0 left edge to 1.0 right edge. "
        "Use an empty signs list if no sign letter is visible."
        + target
    )

async def ask_vlm_about_frame(ctx: Any, prompt: str, *, api_key: str) -> str:
    jpeg = await get_camera_frame(ctx)
    return ask_vlm(jpeg, prompt, api_key=api_key)

async def get_delivered_count(ctx: Any) -> int:
    return len(await delivered_cube_ids(ctx))

async def get_held_cube_info(ctx: Any) -> dict[str, str] | None:
    held = await held_cube_info(ctx)
    return {"entity_id": held[0], "color": held[1]} if held else None

async def perceive(ctx: Any) -> tuple[list[Any], tuple[int, int]]:
    jpeg = await get_camera_frame(ctx)
    image = decode_jpeg(jpeg)
    height, width = image.shape[:2]
    return detect_color_blobs(jpeg), (width, height)

async def set_head(ctx: Any, *, yaw: float | None = None, pitch: float | None = None) -> Any:
    args: dict[str, float] = {}
    if yaw is not None: args["yaw"] = yaw
    if pitch is not None: args["pitch"] = pitch
    return await ctx.invoke("set_head", args, timeout_s=10)

async def move_velocity(ctx: Any, *, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, duration_s: float = 1.0) -> Any:
    return await ctx.invoke("set_velocity", {"vx": vx, "vy": vy, "wz": wz, "duration_s": duration_s}, timeout_s=30)

async def pick_nearest_cube(ctx: Any) -> Any:
    # 에러 수정: 불필요한 백슬래시 제거
    return await ctx.invoke("pick_entity", {"target": {"kind": "entity", "entity_id": "cube"}}, timeout_s=300)

async def place_nearest_zone(ctx: Any) -> Any:
    return await ctx.invoke("place_entity", {}, timeout_s=300)

def result_summary(result: Any) -> dict[str, Any]:
    # 에러 수정: 불필요한 백슬래시 제거
    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    return {
        "status": str(status) if status is not None else None,
        "error": getattr(error, "message", None) if error else None,
    }

async def scan_head(ctx: Any, *, yaws: tuple[float, ...] = (-0.8, 0.0, 0.8), pitch: float = 0.15) -> list[Any]:
    all_detections: list[Any] = []
    for yaw in yaws:
        await set_head(ctx, yaw=yaw, pitch=pitch)
        await asyncio.sleep(0.4)
        detections, frame_size = await perceive(ctx)
        for detection in detections:
            all_detections.append(
                ScannedDetection(
                    color=detection.color,
                    angle_deg=detection.angle_deg,
                    blob_area=detection.blob_area,
                    centroid=detection.centroid,
                    bbox=detection.bbox,
                    frame_size=frame_size,
                    head_yaw=yaw,
                    head_pitch=pitch,
                )
            )
    return all_detections

async def capture_view(ctx: Any, memory: AgentMemory, label: str) -> str | None:
    memory.capture_index += 1
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "view"
    path = CAPTURE_DIR / f"{memory.capture_index:04d}_{safe_label}.jpg"
    try:
        await screenshot(ctx, f"Capture {memory.capture_index}: {label}", path)
    except Exception as exc:
        print(f"Capture failed ({label}): {exc}")
        return None
    return str(path)

def _robot_xy_yaw(robot_status: Any) -> tuple[float, float, float]:
    pose = robot_status.robot.pose
    return pose.position[0], pose.position[1], math.radians(pose.yaw_deg)

def _xy_from_bearing(
    robot_status: Any,
    bearing_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    rx, ry, yaw = _robot_xy_yaw(robot_status)
    absolute_bearing = yaw + math.radians(bearing_deg)
    return (
        rx + distance_m * math.cos(absolute_bearing),
        ry + distance_m * math.sin(absolute_bearing),
    )

def _distance_to_xy(robot_status: Any, xy: tuple[float, float]) -> float:
    rx, ry, _ = _robot_xy_yaw(robot_status)
    return math.hypot(xy[0] - rx, xy[1] - ry)

def _format_xy(xy: tuple[float, float] | None) -> str:
    if xy is None:
        return "None"
    return f"({xy[0]:+.2f}, {xy[1]:+.2f})"

def _absolute_bearing_rad(robot_status: Any, bearing_deg: float) -> float:
    _, _, yaw = _robot_xy_yaw(robot_status)
    return yaw + math.radians(bearing_deg)

def _cross2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]

def _ray_intersection(
    a: SignObservation,
    b: SignObservation,
) -> tuple[tuple[float, float], float] | None:
    ax, ay = a.robot_xy
    bx, by = b.robot_xy
    baseline = math.hypot(bx - ax, by - ay)
    if baseline < MIN_SIGN_BASELINE_M:
        return None

    da = (math.cos(a.absolute_bearing_rad), math.sin(a.absolute_bearing_rad))
    db = (math.cos(b.absolute_bearing_rad), math.sin(b.absolute_bearing_rad))
    denom = _cross2(da, db)
    if abs(denom) < 1e-3:
        return None

    angle = abs(math.degrees(math.atan2(denom, da[0] * db[0] + da[1] * db[1])))
    angle = min(angle, 180.0 - angle)
    if angle < MIN_SIGN_TRIANGULATION_ANGLE_DEG:
        return None

    delta = (bx - ax, by - ay)
    ta = _cross2(delta, db) / denom
    tb = _cross2(delta, da) / denom
    if not (0.4 <= ta <= MAX_SIGN_RAY_DISTANCE_M and 0.4 <= tb <= MAX_SIGN_RAY_DISTANCE_M):
        return None

    confidence = min(a.confidence, b.confidence) * min(1.0, angle / 35.0)
    return ((ax + ta * da[0], ay + ta * da[1]), confidence)

def _too_close_to_rejected(memory: AgentMemory, key: str, xy: tuple[float, float]) -> bool:
    return any(math.hypot(xy[0] - bad[0], xy[1] - bad[1]) < 1.0 for bad in memory.rejected_locations.get(key, []))

def reject_location(memory: AgentMemory, key: str) -> None:
    old = memory.known_locations.pop(key, None)
    memory.location_confidence.pop(key, None)
    memory.sign_observations.pop(key, None)
    if old is not None:
        memory.rejected_locations.setdefault(key, []).append(old)

def triangulate_sign_location(memory: AgentMemory, key: str) -> tuple[tuple[float, float], float] | None:
    observations = memory.sign_observations.get(key, [])
    intersections: list[tuple[tuple[float, float], float]] = []

    for i, first in enumerate(observations):
        for second in observations[i + 1 :]:
            candidate = _ray_intersection(first, second)
            if candidate is not None:
                intersections.append(candidate)

    if not intersections:
        return None

    total_weight = sum(max(weight, 0.05) for _, weight in intersections)
    x = sum(point[0] * max(weight, 0.05) for point, weight in intersections) / total_weight
    y = sum(point[1] * max(weight, 0.05) for point, weight in intersections) / total_weight
    spread = max(math.hypot(point[0] - x, point[1] - y) for point, _ in intersections)
    if spread > MAX_SIGN_ESTIMATE_SPREAD_M:
        return None

    confidence = min(0.95, sum(weight for _, weight in intersections) / max(2, len(intersections)))
    return (x, y), confidence

def record_sign_observation(
    memory: AgentMemory,
    robot_status: Any,
    sighting: SignSighting,
) -> tuple[tuple[float, float], float] | None:
    rx, ry, _ = _robot_xy_yaw(robot_status)
    observation = SignObservation(
        letter=sighting.letter,
        robot_xy=(rx, ry),
        absolute_bearing_rad=_absolute_bearing_rad(robot_status, sighting.bearing_deg),
        confidence=sighting.confidence,
    )
    rows = memory.sign_observations.setdefault(sighting.letter, [])

    # Keep one representative observation per nearby robot pose so repeated
    # head yaws from the same spot do not pretend to be triangulation.
    for index, old in enumerate(rows):
        if math.hypot(old.robot_xy[0] - rx, old.robot_xy[1] - ry) < 0.35:
            if sighting.confidence > old.confidence:
                rows[index] = observation
            break
    else:
        rows.append(observation)

    del rows[:-SIGN_OBSERVATION_LIMIT]
    estimate = triangulate_sign_location(memory, sighting.letter)
    if estimate is None:
        return None

    xy, confidence = estimate
    if _too_close_to_rejected(memory, sighting.letter, xy):
        return None

    remember_location(memory, sighting.letter, xy, confidence=confidence)
    return xy, confidence

def _position_hint_to_bearing(position: str) -> float:
    normalized = position.lower().replace("-", "_").replace(" ", "_")
    if "far_left" in normalized:
        return -35.0
    if "left" in normalized:
        return -22.0
    if "far_right" in normalized:
        return 35.0
    if "right" in normalized:
        return 22.0
    return 0.0

def _x_position_to_bearing(row: dict[str, Any]) -> float | None:
    for key in ("x_position", "x", "center_x", "horizontal_center"):
        raw = row.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 1.0:
            value = value / 100.0
        if 0.0 <= value <= 1.0:
            return (value - 0.5) * 2.0 * SIGN_HFOV_HALF_DEG
    return None

def _json_blob(text: str) -> Any | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None

def parse_sign_sightings(text: str, *, head_yaw: float) -> list[SignSighting]:
    data = _json_blob(text)
    rows: list[Any] = []
    if isinstance(data, dict):
        for key in ("signs", "visible_signs", "spots", "landmarks"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
    elif isinstance(data, list):
        rows = data

    sightings: list[SignSighting] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_letter = row.get("letter") or row.get("sign") or row.get("spot")
        if not isinstance(raw_letter, str):
            continue
        match = re.search(r"[ABCDE]", raw_letter.upper())
        if not match:
            continue
        letter = match.group(0)
        position = str(row.get("position") or row.get("horizontal_position") or row.get("location") or "center")
        confidence_raw = row.get("confidence", 0.65)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.65
        image_bearing = _x_position_to_bearing(row)
        if image_bearing is None:
            image_bearing = _position_hint_to_bearing(position)
        bearing = image_bearing + math.degrees(head_yaw)
        sightings.append(
            SignSighting(
                letter=letter,
                bearing_deg=bearing,
                confidence=max(0.0, min(confidence, 1.0)),
                head_yaw=head_yaw,
                position_hint=position,
                raw=str(row)[:160],
            )
        )

    if sightings:
        return sightings

    # Backup parser for models that answer in prose instead of JSON.
    for letter in sorted(set(re.findall(r"\b[ABCDE]\b", text.upper()))):
        sightings.append(SignSighting(letter=letter, bearing_deg=math.degrees(head_yaw), confidence=0.35, head_yaw=head_yaw, raw=text[:160]))
    return sightings

def remember_location(
    memory: AgentMemory,
    key: str,
    xy: tuple[float, float],
    *,
    confidence: float,
) -> None:
    old = memory.known_locations.get(key)
    old_conf = memory.location_confidence.get(key, 0.0)
    if old is None or confidence >= old_conf:
        blend = 0.0 if old is None or confidence >= 0.95 else 0.35
        memory.known_locations[key] = (
            old[0] * blend + xy[0] * (1.0 - blend) if old else xy[0],
            old[1] * blend + xy[1] * (1.0 - blend) if old else xy[1],
        )
        memory.location_confidence[key] = max(confidence, old_conf)

async def scan_signage(
    ctx: Any,
    memory: AgentMemory,
    robot_status: Any,
    *,
    needed_letter: str | None = None,
    yaws: tuple[float, ...] = SIGN_SCAN_YAWS,
    label_prefix: str = "sign_scan",
) -> list[SignSighting]:
    api_key = getattr(getattr(ctx, "config", None), "tokamak_api_key", "")
    if not api_key:
        try:
            api_key = load_config(require_tokamak=False).tokamak_api_key
        except Exception:
            api_key = ""
    if not api_key:
        return []

    sightings: list[SignSighting] = []
    for yaw in yaws:
        await set_head(ctx, yaw=yaw, pitch=0.05)
        await asyncio.sleep(0.25)
        await capture_view(ctx, memory, f"{label_prefix}_{needed_letter or 'any'}_yaw_{yaw:+.2f}")
        try:
            reply = await ask_vlm_about_frame(
                ctx,
                build_signage_vlm_prompt(needed_letter),
                api_key=api_key,
            )
        except Exception as exc:
            memory.last_vlm_summary = f"VLM failed: {exc}"
            continue

        memory.last_vlm_summary = reply[:500]
        parsed = parse_sign_sightings(reply, head_yaw=yaw)
        if parsed:
            print(
                "Sign scan "
                f"{label_prefix} yaw={yaw:+.2f}: "
                + ", ".join(f"{item.letter}@{item.bearing_deg:+.1f}deg conf={item.confidence:.2f}" for item in parsed)
            )
        else:
            print(f"Sign scan {label_prefix} yaw={yaw:+.2f}: no parsed signs | VLM={reply[:120]!r}")
        for sighting in parsed:
            sightings.append(sighting)
            estimate = record_sign_observation(memory, robot_status, sighting)
            count = len(memory.sign_observations.get(sighting.letter, []))
            if estimate is None:
                print(f"  {sighting.letter}: observation_count={count}, waiting for triangulation")
            else:
                xy, confidence = estimate
                print(f"  {sighting.letter}: triangulated at {_format_xy(xy)} confidence={confidence:.2f}")

    return sightings

async def half_turn_for_search(ctx: Any, memory: AgentMemory, target_letter: str | None) -> dict[str, Any]:
    result = await move_velocity(ctx, vx=0.2, wz=0.5, duration_s=6.3)
    await capture_view(ctx, memory, f"half_turn_search_{target_letter or 'target'}")
    return result_summary(result)

def _contains_sign(sightings: list[SignSighting], letter: str | None) -> bool:
    return letter is not None and any(s.letter == letter for s in sightings)

def _observation_contains_sign(observation: Observation, letter: str | None) -> bool:
    return letter is not None and any(sign.letter == letter for sign in observation.signs)

async def search_landmark_with_turnaround(
    ctx: Any,
    memory: AgentMemory,
    *,
    target_letter: str,
) -> list[SignSighting]:
    robot_status = await get_robot_status(ctx)
    first_sightings = await scan_signage(
        ctx,
        memory,
        robot_status,
        needed_letter=target_letter,
        yaws=TARGET_SCAN_YAWS,
        label_prefix="look_left_right",
    )
    if target_letter in memory.known_locations:
        return first_sightings

    target_sighting = _best_target_sighting(first_sightings, target_letter)
    if target_sighting is not None:
        memory.search_turns += 1
        viewpoint_xy = _triangulation_viewpoint_xy(robot_status, target_sighting, memory.search_turns)
        print(
            f"{target_letter} was visible but not triangulated yet; "
            f"moving laterally 90deg from bearing {target_sighting.bearing_deg:+.1f}deg "
            f"to second viewpoint {_format_xy(viewpoint_xy)} before turning around."
        )
        result = await go_to_xy(ctx, *viewpoint_xy)
        print(f"Second viewpoint move result: {result_summary(result)}")
        await capture_view(ctx, memory, f"second_viewpoint_for_{target_letter}")
        robot_status = await get_robot_status(ctx)
        viewpoint_sightings = await scan_signage(
            ctx,
            memory,
            robot_status,
            needed_letter=target_letter,
            yaws=TARGET_SCAN_YAWS,
            label_prefix="second_viewpoint_left_right",
        )
        if target_letter in memory.known_locations:
            return first_sightings + viewpoint_sightings
        first_sightings = first_sightings + viewpoint_sightings

        target_sighting = _best_target_sighting(viewpoint_sightings, target_letter)
        if target_sighting is not None:
            memory.search_turns += 1
            opposite_xy = _triangulation_viewpoint_xy(robot_status, target_sighting, memory.search_turns)
            print(
                f"{target_letter} is still visible but triangulation is weak; "
                f"trying opposite lateral viewpoint {_format_xy(opposite_xy)}."
            )
            result = await go_to_xy(ctx, *opposite_xy)
            print(f"Opposite viewpoint move result: {result_summary(result)}")
            await capture_view(ctx, memory, f"opposite_viewpoint_for_{target_letter}")
            robot_status = await get_robot_status(ctx)
            opposite_sightings = await scan_signage(
                ctx,
                memory,
                robot_status,
                needed_letter=target_letter,
                yaws=TARGET_SCAN_YAWS,
                label_prefix="opposite_viewpoint_left_right",
            )
            if target_letter in memory.known_locations:
                return first_sightings + opposite_sightings
            first_sightings = first_sightings + opposite_sightings

    await half_turn_for_search(ctx, memory, target_letter)
    robot_status = await get_robot_status(ctx)
    second_sightings = await scan_signage(
        ctx,
        memory,
        robot_status,
        needed_letter=target_letter,
        yaws=TARGET_SCAN_YAWS,
        label_prefix="after_half_turn_left_right",
    )
    return first_sightings + second_sightings

# ---------------------------------------------------------------------------
# [학생 TODO 1] 시각 기반 정밀 월드 좌표 추정 함수 (Mathematical Localization)
# ---------------------------------------------------------------------------
def _camera_height_m(robot_status: Any) -> float:
    position = getattr(getattr(robot_status, "robot", None), "pose", None)
    raw_position = getattr(position, "position", None)
    if raw_position is not None and len(raw_position) >= 3:
        try:
            height = float(raw_position[2])
        except (TypeError, ValueError):
            height = DEFAULT_CAMERA_HEIGHT_M
        if 0.7 <= height <= 2.2:
            return height
    return DEFAULT_CAMERA_HEIGHT_M

def _xy_from_body_offset(robot_status: Any, forward_m: float, left_m: float) -> tuple[float, float]:
    rx, ry, yaw = _robot_xy_yaw(robot_status)
    return (
        rx + forward_m * math.cos(yaw) + left_m * math.cos(yaw + math.pi / 2),
        ry + forward_m * math.sin(yaw) + left_m * math.sin(yaw + math.pi / 2),
    )

def _ground_plane_xy_from_detection(
    observation: Observation,
    detection: ScannedDetection,
) -> tuple[tuple[float, float], float] | None:
    width, height = detection.frame_size
    if width <= 0 or height <= 0:
        return None

    _, y, _, bbox_h = detection.bbox
    y_bottom = min(height - 1, max(0, y + bbox_h))
    fx = (width / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG / 2.0))
    fy = fx
    alpha = math.atan2(y_bottom - height / 2.0, fy)
    downward_angle = detection.head_pitch + CAMERA_PITCH_OFFSET_RAD + alpha
    if downward_angle <= math.radians(4.0):
        return None

    forward_m = _camera_height_m(observation.robot_status) / math.tan(downward_angle)
    if not (GROUND_DISTANCE_MIN_M <= forward_m <= GROUND_DISTANCE_MAX_M):
        return None

    bearing_rad = math.radians(detection.full_bearing_deg)
    left_m = forward_m * math.tan(bearing_rad)
    xy = _xy_from_body_offset(observation.robot_status, forward_m, left_m)
    return xy, forward_m

def _blob_size_xy_from_detection(
    observation: Observation,
    detection: ScannedDetection,
) -> tuple[float, float] | None:
    if detection.blob_area <= 0:
        return None
    distance = CUBE_DISTANCE_K / math.sqrt(detection.blob_area)
    distance = max(0.55, min(distance, 5.5))
    return _xy_from_bearing(observation.robot_status, detection.full_bearing_deg, distance)

def _candidate_detections_for_target(
    observation: Observation,
    target_color: str,
) -> list[ScannedDetection]:
    if target_color in {"A", "cube", "any"}:
        return observation.detections
    pad_color = SIGN_TO_PAD_COLOR.get(target_color)
    if pad_color is not None:
        return [d for d in observation.detections if d.color == pad_color]
    return [d for d in observation.detections if d.color == target_color]

def _select_detection_for_target(
    observation: Observation,
    target_color: str,
    targets: list[ScannedDetection],
) -> ScannedDetection:
    if target_color in SIGN_TO_PAD_COLOR:
        sign = _best_target_sighting(observation.signs, target_color)
        if sign is not None:
            close_to_sign = [
                detection
                for detection in targets
                if abs(detection.full_bearing_deg - sign.bearing_deg) <= 35.0
            ]
            if close_to_sign:
                return min(
                    close_to_sign,
                    key=lambda detection: (
                        abs(detection.full_bearing_deg - sign.bearing_deg),
                        -detection.blob_area,
                    ),
                )
    return max(targets, key=lambda detection: detection.blob_area)

def estimate_target_xy_from_observation(observation: Observation, target_color: str | None) -> tuple[float, float] | None:
    if not target_color:
        return None

    targets = _candidate_detections_for_target(observation, target_color)
    if not targets:
        return None

    matched_target = _select_detection_for_target(observation, target_color, targets)
    ground_estimate = _ground_plane_xy_from_detection(observation, matched_target)
    if ground_estimate is not None:
        xy, distance = ground_estimate
        print(
            f"Ground-plane estimate for {target_color}: "
            f"xy={_format_xy(xy)} distance={distance:.2f}m "
            f"bbox={matched_target.bbox} frame={matched_target.frame_size}"
        )
        return xy

    xy = _blob_size_xy_from_detection(observation, matched_target)
    if xy is not None:
        print(
            f"Blob-size fallback estimate for {target_color}: "
            f"xy={_format_xy(xy)} area={matched_target.blob_area}"
        )
    return xy

def _nearest_visible_blob_area(observation: Observation, color: str | None = None) -> int:
    candidates = observation.detections if color is None else [d for d in observation.detections if d.color == color]
    if not candidates:
        return 0
    return max(int(d.blob_area) for d in candidates)

def _source_cube_evidence(observation: Observation) -> tuple[int, int]:
    large_blobs = [detection for detection in observation.detections if int(detection.blob_area) >= SOURCE_CUBE_AREA_THRESHOLD]
    return len(large_blobs), _nearest_visible_blob_area(observation)

def maybe_remember_source_at_current_pose(memory: AgentMemory, observation: Observation) -> bool:
    saw_a = _observation_contains_sign(observation, "A")
    large_count, max_area = _source_cube_evidence(observation)
    if saw_a and (large_count >= SOURCE_CUBE_COUNT_THRESHOLD or max_area >= 7000):
        rx, ry, _ = _robot_xy_yaw(observation.robot_status)
        remember_location(memory, "A", (rx, ry), confidence=0.9)
        print(
            "A source accepted at current pose "
            f"{_format_xy((rx, ry))}: saw A and cube evidence "
            f"(large_blobs={large_count}, max_area={max_area})."
        )
        return True
    return False

def _relative_explore_xy(robot_status: Any, search_turn: int) -> tuple[float, float]:
    # Body-frame waypoints. go_to handles path planning, so these can reveal
    # signs hidden behind shelves/walls better than spinning in place.
    pattern = (
        (2.6, 0.0),
        (1.8, 1.6),
        (1.8, -1.6),
        (3.2, 0.9),
        (3.2, -0.9),
        (-0.8, 1.8),
    )
    forward, left = pattern[search_turn % len(pattern)]
    rx, ry, yaw = _robot_xy_yaw(robot_status)
    return (
        rx + forward * math.cos(yaw) + left * math.cos(yaw + math.pi / 2),
        ry + forward * math.sin(yaw) + left * math.sin(yaw + math.pi / 2),
    )

def _best_target_sighting(sightings: list[SignSighting], target_letter: str) -> SignSighting | None:
    candidates = [sighting for sighting in sightings if sighting.letter == target_letter]
    if not candidates:
        return None
    return min(candidates, key=lambda sighting: (abs(sighting.bearing_deg), -sighting.confidence))

def _triangulation_viewpoint_xy(
    robot_status: Any,
    sighting: SignSighting,
    search_turn: int,
) -> tuple[float, float]:
    rx, ry, yaw = _robot_xy_yaw(robot_status)
    target_bearing = yaw + math.radians(sighting.bearing_deg)
    side = TRIANGULATION_SIDE_M if search_turn % 2 == 0 else -TRIANGULATION_SIDE_M
    perpendicular = target_bearing + math.pi / 2
    return (
        rx + TRIANGULATION_FORWARD_M * math.cos(target_bearing) + side * math.cos(perpendicular),
        ry + TRIANGULATION_FORWARD_M * math.sin(target_bearing) + side * math.sin(perpendicular),
    )

# ---------------------------------------------------------------------------
# [학생 TODO 2] 고급 LLM 의사결정 자율 루프 (ReAct 파이프라인)
# ---------------------------------------------------------------------------
ACTION_DESCRIPTIONS = {
    "search_cube": "Look for the A source area or visible cubes. Use when no reliable source/cube target exists.",
    "navigate_to_cube": "Move toward the remembered A source area or a visually estimated cube coordinate.",
    "pick_cube": "Pick the nearest cube. Use only when a cube is close enough or the robot is near A.",
    "search_pad": "Search for the destination sign matching the held cube. Use when that pad coordinate is unknown or rejected.",
    "navigate_to_pad": "Move to a triangulated/remembered destination pad coordinate.",
    "place_cube": "Place the held cube. Use only near the matching destination and after visual confirmation.",
    "recover": "Back up or change viewpoint after failed navigation, target loss, or unsafe state.",
    "skip_target": "Abandon the current unreliable target and search again.",
    "stop": "Stop only when the task is complete or continuing is unsafe/impossible.",
}

def fallback_decision(observation: Observation, memory: AgentMemory) -> AgentDecision:
    if not memory.held_color:
        if "A" in memory.known_locations and _distance_to_xy(observation.robot_status, memory.known_locations["A"]) <= NEAR_SOURCE_M:
            return AgentDecision(next_action="pick_cube", target_color="A", reason="Fallback: near A source; pick nearest cube.")
        if _observation_contains_sign(observation, "A") and _source_cube_evidence(observation)[0] >= SOURCE_CUBE_COUNT_THRESHOLD:
            return AgentDecision(next_action="pick_cube", target_color="cube", reason="Fallback: A sign and nearby cube evidence indicate source is reached.")
        if _nearest_visible_blob_area(observation) >= 7000:
            return AgentDecision(next_action="pick_cube", target_color="cube", reason="Fallback: a cube-sized blob is very close.")
        if "A" in memory.known_locations:
            return AgentDecision(next_action="navigate_to_cube", target_color="A", reason="Fallback: navigate to remembered A source.")
        if observation.detections:
            return AgentDecision(next_action="navigate_to_cube", target_color="cube", reason="Fallback: cubes are visible; approach nearest cube.")
        return AgentDecision(next_action="search_cube", target_color="A", reason="Fallback: search for A source or cubes.")

    target_letter = DESTINATION_SIGN_RULES.get(memory.held_color)
    if target_letter is None:
        return AgentDecision(next_action="recover", reason=f"Fallback: unknown held color {memory.held_color!r}.")
    if (
        target_letter in memory.known_locations
        and _distance_to_xy(observation.robot_status, memory.known_locations[target_letter]) <= NEAR_PAD_M
        and _observation_contains_sign(observation, target_letter)
    ):
        return AgentDecision(next_action="place_cube", target_color=target_letter, reason="Fallback: matching pad is nearby and visible.")
    if target_letter in memory.known_locations and _distance_to_xy(observation.robot_status, memory.known_locations[target_letter]) <= NEAR_PAD_M:
        reject_location(memory, target_letter)
        return AgentDecision(next_action="search_pad", target_color=target_letter, reason="Fallback: estimated pad is nearby but not visible; reject and rescan.")
    if target_letter in memory.known_locations:
        return AgentDecision(next_action="navigate_to_pad", target_color=target_letter, reason="Fallback: navigate to triangulated destination pad.")
    return AgentDecision(next_action="search_pad", target_color=target_letter, reason="Fallback: search for destination sign.")

async def decide_next_action(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    context = build_decision_context(task, observation, memory, last_result)
    context["available_actions"] = ACTION_DESCRIPTIONS
    context["routing_rules"] = {
        "source": "A",
        "red": "B",
        "green": "C",
        "blue": "D",
        "yellow": "E",
    }
    large_count, max_area = _source_cube_evidence(observation)
    context["source_evidence"] = {
        "saw_A_sign": _observation_contains_sign(observation, "A"),
        "large_cube_blob_count": large_count,
        "max_cube_blob_area": max_area,
        "source_location_known": "A" in memory.known_locations,
    }
    context["safety_rules"] = [
        "Do not choose place_cube unless the held cube's matching sign is visible nearby.",
        "If a destination coordinate is unknown, choose search_pad before navigate_to_pad.",
        "If a sign was seen but not triangulated, search_pad is preferred so the controller can gather another viewpoint.",
        "For source A, if saw_A_sign is true and cube blobs are close, choose pick_cube rather than repeatedly searching.",
        "If not holding a cube, choose search_cube, navigate_to_cube, or pick_cube.",
        "If holding a cube, choose search_pad, navigate_to_pad, place_cube, or recover.",
    ]

    api_key = memory.tokamak_api_key
    if not api_key:
        try:
            api_key = load_config(require_tokamak=False).tokamak_api_key
        except Exception:
            api_key = ""
    if not api_key:
        decision = fallback_decision(observation, memory)
        decision.reason = "No TOKAMAK_API_KEY; " + decision.reason
        return decision

    system_prompt = (
        "You are the high-level decision maker for a Level 1 warehouse robot sorting task. "
        "Choose exactly one next_action from the provided available_actions. "
        "You do not output low-level velocity commands. The controller will execute your high-level action. "
        "Return ONLY JSON with this schema: "
        '{"next_action":"search_pad","target_color":"C","reason":"short reason","recovery_strategy":null}. '
        "target_color may be a cube color, 'cube', or a sign letter A/B/C/D/E depending on the action."
    )

    try:
        reply = call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            api_key=api_key,
            timeout_s=45,
        )
        decision = parse_agent_decision(reply)
        if decision is not None:
            decision.reason = "LLM: " + decision.reason
            return decision
        print(f"LLM decision parse failed: {reply[:300]!r}")
    except Exception as exc:
        print(f"LLM decision call failed: {exc}")

    decision = fallback_decision(observation, memory)
    decision.reason = "LLM unavailable/invalid; " + decision.reason
    return decision

# ---------------------------------------------------------------------------
# [학생 TODO 3] 시계열 관찰값 수집 및 자동 매핑 가속화
# ---------------------------------------------------------------------------
async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    robot_status = await get_robot_status(ctx)
    scanned_detections = await scan_head(ctx, yaws=COLOR_SCAN_YAWS)
    needed_letter = None
    if memory.held_color:
        needed_letter = DESTINATION_SIGN_RULES.get(memory.held_color)
    elif "A" not in memory.known_locations:
        needed_letter = "A"

    signs: list[SignSighting] = []
    should_scan_sign = needed_letter is not None and needed_letter not in memory.known_locations
    if (
        memory.held_color
        and needed_letter in memory.known_locations
        and _distance_to_xy(robot_status, memory.known_locations[needed_letter]) <= NEAR_PAD_M * 1.8
    ):
        should_scan_sign = True
    if should_scan_sign:
        signs = await scan_signage(
            ctx,
            memory,
            robot_status,
            needed_letter=needed_letter,
            yaws=TARGET_SCAN_YAWS if memory.held_color else SIGN_SCAN_YAWS,
            label_prefix="observe_target_sign" if memory.held_color else "observe_source_sign",
        )

    current_obs = Observation(
        robot_status=robot_status,
        detections=scanned_detections,
        signs=signs,
        note="",
        vlm_summary=memory.last_vlm_summary,
    )
    if not memory.held_color:
        maybe_remember_source_at_current_pose(memory, current_obs)
    if (
        needed_letter in SIGN_TO_PAD_COLOR
        and _observation_contains_sign(current_obs, needed_letter)
        and needed_letter not in memory.known_locations
    ):
        pad_xy = estimate_target_xy_from_observation(current_obs, needed_letter)
        if pad_xy is not None and not _too_close_to_rejected(memory, needed_letter, pad_xy):
            remember_location(memory, needed_letter, pad_xy, confidence=0.96)
            print(
                f"{needed_letter} pad accepted from ground-plane camera estimate "
                f"at {_format_xy(pad_xy)}."
            )

    note = (
        f"known={sorted(memory.known_locations)} "
        f"needed={needed_letter or '-'}"
    )
    current_obs.note = note
    return current_obs

# ---------------------------------------------------------------------------
# [학생 TODO 4 & 5] 결과 물리 검증 및 지속 상태 관리 인터페이스
# ---------------------------------------------------------------------------
async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(0.2)
    held_info = await get_held_cube_info(ctx)
    delivered_val = await get_delivered_count(ctx)
    
    return {
        "decision": decision.__dict__,
        "action_result": action_result,
        "delivered_count": delivered_val,
        "held_cube": held_info,
        "held_color": held_info["color"] if held_info else None,
    }

def update_memory(memory: AgentMemory, observation: Observation, decision: AgentDecision, verified: dict[str, Any]) -> None:
    if "delivered_count" in verified:
        memory.delivered_count = int(verified["delivered_count"])
    
    memory.held_color = verified.get("held_color")
    
    # 스테이지 상태 머신 전이 트래킹
    if memory.held_color:
        memory.stage = "holding_cube"
        memory.active_color = memory.held_color
    else:
        memory.stage = "need_cube"
        memory.active_color = None

    memory.logs.append({
        "observation": {
            "visible_count": len(observation.detections),
            "visible_signs": [sign.letter for sign in observation.signs],
            "known_locations": {key: tuple(round(v, 2) for v in xy) for key, xy in sorted(memory.known_locations.items())},
            "sign_observation_counts": {key: len(value) for key, value in sorted(memory.sign_observations.items())},
            "rejected_locations": {key: len(value) for key, value in sorted(memory.rejected_locations.items())},
            "delivered_count": memory.delivered_count,
            "held_color": memory.held_color,
            "stage": memory.stage
        },
        "llm_decision": decision.__dict__,
        "verified": verified,
    })

# ---------------------------------------------------------------------------
# [학생 TODO 6] 좌표 유도 주행부 및 의사결정 하위 매핑 실행기
# ---------------------------------------------------------------------------
async def go_to_xy(ctx: Any, x: float, y: float) -> Any:
    return await ctx.invoke(
        "go_to",
        {
            "target": {
                "kind": "pose",
                "pose": {"frame_id": "world", "position": [x, y, 0]},
            }
        },
        timeout_s=300,
    )

async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    if decision.next_action in {"search_cube", "search_pad"}:
        needed_letter = decision.target_color if decision.target_color in LANDMARK_LETTERS else None
        if needed_letter:
            sightings = await search_landmark_with_turnaround(ctx, memory, target_letter=needed_letter)
        else:
            sightings = await scan_signage(ctx, memory, observation.robot_status, needed_letter=needed_letter)
        if needed_letter and needed_letter in memory.known_locations:
            return {
                "action": decision.next_action,
                "status": "landmark_found",
                "target": needed_letter,
                "target_xy": memory.known_locations[needed_letter],
                "reason": "target sign has enough observations for triangulated coordinate",
            }
        if decision.next_action == "search_cube" and observation.detections:
            return {"action": decision.next_action, "status": "cube_blob_visible", "visible_count": len(observation.detections)}

        memory.search_turns += 1
        current_status = await get_robot_status(ctx)
        explore_xy = _relative_explore_xy(current_status, memory.search_turns)
        result = await go_to_xy(ctx, *explore_xy)
        await capture_view(ctx, memory, f"{decision.next_action}_explore_viewpoint")
        return {
            "action": decision.next_action,
            "status": "explored_new_viewpoint",
            "target": needed_letter,
            "sightings": [s.letter for s in sightings],
            "target_seen": _contains_sign(sightings, needed_letter),
            "target_observation_count": len(memory.sign_observations.get(needed_letter, [])) if needed_letter else 0,
            "reason": "target coordinate not triangulated yet, moved to new exploration viewpoint",
            "explore_xy": explore_xy,
            "result": result_summary(result),
        }

    if decision.next_action in {"navigate_to_cube", "navigate_to_pad"}:
        target_xy = memory.known_locations.get(decision.target_color)
        if not target_xy:
            target_xy = estimate_target_xy_from_observation(observation, decision.target_color)

        if target_xy is None:
            memory.search_turns += 1
            current_status = await get_robot_status(ctx)
            explore_xy = _relative_explore_xy(current_status, memory.search_turns)
            result = await go_to_xy(ctx, *explore_xy)
            await capture_view(ctx, memory, f"{decision.next_action}_failed_then_explored")
            return {
                "action": decision.next_action,
                "status": "failed_then_explored",
                "reason": "no coordinate estimate",
                "explore_xy": explore_xy,
                "result": result_summary(result),
            }

        result = await go_to_xy(ctx, *target_xy)
        await capture_view(ctx, memory, f"{decision.next_action}_{decision.target_color or 'target'}")
        if decision.target_color in LANDMARK_LETTERS:
            remember_location(memory, decision.target_color, target_xy, confidence=memory.location_confidence.get(decision.target_color, 0.6))
        return {"action": decision.next_action, "target_xy": target_xy, "result": result_summary(result)}

    if decision.next_action == "pick_cube":
        result = await pick_nearest_cube(ctx)
        await capture_view(ctx, memory, "after_pick_cube")
        return {"action": "pick_cube", "result": result_summary(result)}

    if decision.next_action == "place_cube":
        result = await place_nearest_zone(ctx)
        await capture_view(ctx, memory, "after_place_cube")
        return {"action": "place_cube", "result": result_summary(result)}

    if decision.next_action == "recover":
        memory.search_turns += 1
        await move_velocity(ctx, vx=-0.25, duration_s=1.0)
        await capture_view(ctx, memory, "recover_back_step")
        explore_xy = _relative_explore_xy(observation.robot_status, memory.search_turns)
        result = await go_to_xy(ctx, *explore_xy)
        await capture_view(ctx, memory, "recover_replanned_viewpoint")
        return {"action": "recover", "status": "backed_up_and_replanned", "explore_xy": explore_xy, "result": result_summary(result)}

    return {"action": decision.next_action, "status": "no_op"}

# ---------------------------------------------------------------------------
# 메인 제어 루프 드라이버
# ---------------------------------------------------------------------------
async def run_agent(
    ctx: Any,
    *,
    max_cycles: int = 10_000,
    completion: CompletionConfig | None = None,
) -> AgentMemory:
    memory = AgentMemory()
    memory.tokamak_api_key = getattr(getattr(ctx, "config", None), "tokamak_api_key", "")
    last_result: dict[str, Any] | None = None
    tracker = CompletionTracker(completion) if completion is not None else None

    for cycle in range(1, max_cycles + 1):
        print(f"\n[Level 1 Adaptive Run] Cycle {cycle} | Score 누적: {memory.delivered_count * 20}점")
        if tracker is not None:
            first_cycle = tracker.started_at is None
            tracker.start_first_cycle()
            if first_cycle:
                tracker.print_start()
            reason = await tracker.stop_reason_from_scene(ctx)
            if reason is not None:
                tracker.mark_ended(reason)
                break

        # 1. 월드 인지 및 상태 업데이트
        observation = await observe_world(ctx, memory)
        
        # 2. LLM / Fallback을 통한 의사결정 추론
        decision = await decide_next_action(TASK, observation, memory, last_result)
        print(f"Observation note: {observation.note}")
        print(f"Known locations: { {k: tuple(round(v, 2) for v in xy) for k, xy in sorted(memory.known_locations.items())} }")
        print(f"Sign observation counts: { {k: len(v) for k, v in sorted(memory.sign_observations.items())} }")
        if last_result is not None:
            print(f"Last result summary: {last_result}")
        print(f"-> Selected Action: {decision.next_action} (Target: {decision.target_color})")
        print(f"-> Reason: {decision.reason}")

        if decision.next_action == "stop":
            break

        # 3. 행동 실행기 구동
        action_result = await execute_decision(ctx, decision, observation, memory)
        print(f"Action result: {action_result}")
        action_capture = await capture_view(ctx, memory, f"cycle_{cycle:03d}_after_{decision.next_action}")
        if action_capture is not None:
            action_result["capture_path"] = action_capture
        
        # 4. 물리 피드백 및 검증 정보 래핑
        verified = await verify_outcome(ctx, decision, action_result)
        
        # 5. 메모리 힙 데이터 갱신
        update_memory(memory, observation, decision, verified)
        last_result = verified
        
        if tracker is not None:
            reason = await tracker.stop_reason_from_scene(ctx)
            if reason is not None:
                tracker.mark_ended(reason)
                break

    if tracker is not None:
        await tracker.print_summary_from_scene(ctx)
    return memory


async def run(ctx: Any) -> None:
    print(TASK)
    print("Running Level 1 adaptive-navigation project starter")
    memory = await run_agent(ctx)
    print("\nRun complete.")
    print(f"Delivered count: {memory.delivered_count}")
    print("Logs:")
    for item in memory.logs:
        print(item)
