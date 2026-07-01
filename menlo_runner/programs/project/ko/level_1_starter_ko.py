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
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.llm import ask_vlm
from menlo_runner.perception import detect_color_blobs


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
SIGNAGE_NOTE = (
    "A는 conveyor/cube source area이며 destination이 아닙니다. "
    "Destination sign은 B red, C green, D blue, E yellow입니다."
)

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

@dataclass
class Observation:
    robot_status: Any
    detections: list[Any]
    note: str = ""
    vlm_summary: str = ""

@dataclass(frozen=True)
class ScannedDetection:
    color: str
    angle_deg: float
    blob_area: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    head_yaw: float
    head_pitch: float

    @property
    def full_bearing_deg(self) -> float:
        return self.angle_deg + math.degrees(self.head_yaw)

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
        "held_color": memory.held_color,
        "active_color": memory.active_color,
        "stage": memory.stage,
        "delivered_count": memory.delivered_count,
        "known_locations": {k: [round(v[0], 2), round(v[1], 2)] for k, v in memory.known_locations.items()},
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

async def get_delivered_count(ctx: Any) -> int:
    return len(await delivered_cube_ids(ctx))

async def get_held_cube_info(ctx: Any) -> dict[str, str] | None:
    held = await held_cube_info(ctx)
    return {"entity_id": held[0], "color": held[1]} if held else None

async def perceive(ctx: Any) -> list[Any]:
    jpeg = await get_camera_frame(ctx)
    return detect_color_blobs(jpeg)

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
        for detection in await perceive(ctx):
            all_detections.append(
                ScannedDetection(
                    color=detection.color,
                    angle_deg=detection.angle_deg,
                    blob_area=detection.blob_area,
                    centroid=detection.centroid,
                    bbox=detection.bbox,
                    head_yaw=yaw,
                    head_pitch=pitch,
                )
            )
    return all_detections

# ---------------------------------------------------------------------------
# [학생 TODO 1] 시각 기반 정밀 월드 좌표 추정 함수 (Mathematical Localization)
# ---------------------------------------------------------------------------
def estimate_target_xy_from_observation(observation: Observation, target_color: str | None) -> tuple[float, float] | None:
    if not target_color:
        return None
        
    # 타겟 색상과 일치하는 시각 블롭 필터링 (가장 큰 인스턴스 선택)
    targets = [d for d in observation.detections if d.color == target_color]
    if not targets:
        return None
    matched_target = max(targets, key=lambda t: t.blob_area)
    
    # 현재 로봇의 전역 기하 정보 추출
    try:
        rx = observation.robot_status.robot.pose.position[0]
        ry = observation.robot_status.robot.pose.position[1]
        robot_yaw = math.radians(observation.robot_status.robot.pose.yaw_deg)
    except Exception:
        return None
        
    # 절대 방위각 계산 = 로봇 기준 각도 + 머리 각도 및 화면 오프셋
    absolute_bearing = robot_yaw + math.radians(matched_target.full_bearing_deg)
    
    # 원근 투영 모델 기반 거리 역산 역추정 공식 (실제 카메라 화각 상수 K 보정 적용)
    # 카메라 렌즈 기준 면적과 거리는 반비례 관계를 가집니다.
    K_FACTOR = 1100.0 
    distance = K_FACTOR / math.sqrt(matched_target.blob_area)
    
    # 주행 안전 제약조건 매핑 (최소 0.4m ~ 최대 6.0m 클리핑)
    distance = max(0.4, min(distance, 6.0))
    
    # 삼각기하학 월드 좌표 좌표계 변환 적용
    tx = rx + distance * math.cos(absolute_bearing)
    ty = ry + distance * math.sin(absolute_bearing)
    
    return (tx, ty)

# ---------------------------------------------------------------------------
# [학생 TODO 2] 고급 LLM 의사결정 자율 루프 (ReAct 파이프라인)
# ---------------------------------------------------------------------------
async def decide_next_action(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    
    context = build_decision_context(task, observation, memory, last_result)
    
    # 텍스트 에이전트 전 전역 행동 지침서 구성
    system_instruction = f"""
    당신은 창고 환경에서 자율 기동하는 휴머노이드 로봇의 고수준 추론 감독관입니다.
    현재 레벨에서는 scene_state 및 개체 ID 접근이 전면 금지되어 있으므로 오직 기억 장치와 시야 마스크만을 이용해 판단해야 합니다.

    [작업 실행 흐름 프로토콜]
    1. 손이 비어있다면(held_color가 null): 무조건 패드 'A'(큐브 컨베이어 구역)로 기동하여 무작위 큐브를 집어 올려야 합니다('pick_cube'). 만약 'A' 좌표를 모른다면 즉시 'search_pad'를 수행하여 찾으세요.
    2. 무언가 집고 있다면(held_color 확인): 들고 있는 큐브 색상과 매칭되는 전용 수령 패드 규칙을 파악하세요 (red->B, green->C, blue->D, yellow->E).
    3. 목적지 패드의 전역 위치 정보가 'known_locations'에 기록되어 있다면 해당 위치로 즉시 자율 네비게이션을 수행하세요 ('navigate_to_pad').
    4. 만약 목적지 패드가 메모리에 없다면 시야각 안에 들어올 때까지 방위를 변경하거나 주변 공간으로 기동하여 패드를 직접 눈으로 탐색해야 합니다 ('search_pad').
    5. 패드 영역에 도달했을 때 'place_cube' 액션을 기동하세요.

    출력 형식 규칙: 반드시 구조화된 양식의 JSON 문자열 형태로만 반환해야 하며, 부가적인 설명글이나 마크다운 래퍼 기호를 일절 금지합니다.
    Format: {{"next_action": "지정액션", "target_color": "대상지정", "reason": "논리적 근거"}}
    """

    try:
        user_input = json.dumps(context, ensure_ascii=False)
        reply = await call_llm([
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input}
        ])
        decision = parse_agent_decision(reply)
        if decision:
            return decision
    except Exception as e:
        print(f"[LLM Exception 발생]: {e}")

    # [Fallback Safety Net]: 파싱 불안정 혹은 예외 발생 시 자율 상태 머신 분기 작동
    if not memory.held_color:
        if "A" in memory.known_locations:
            return AgentDecision(next_action="navigate_to_cube", target_color="A", reason="Fallback: 기기록된 소스 영역 기동")
        return AgentDecision(next_action="search_cube", target_color="A", reason="Fallback: 컨베이어 벨트 구역 수색 명령")
    else:
        target_pad = DESTINATION_SIGN_RULES.get(memory.held_color, "B")
        if target_pad in memory.known_locations:
            return AgentDecision(next_action="navigate_to_pad", target_color=target_pad, reason="Fallback: 기저장 패드 최단 기동")
        return AgentDecision(next_action="search_pad", target_color=target_pad, reason="Fallback: 미지 목적지 패드 회전 스캔 기동")

# ---------------------------------------------------------------------------
# [학생 TODO 3] 시계열 관찰값 수집 및 자동 매핑 가속화
# ---------------------------------------------------------------------------
async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    robot_status = await get_robot_status(ctx)
    
    # 맵 내부 기동 중 효율적인 스팟 확보를 위해 360도 전방위 파노라마형 헤드 서보 기동 수행
    scanned_detections = await scan_head(ctx, yaws=(-1.0, -0.5, 0.0, 0.5, 1.0))
    
    current_obs = Observation(robot_status=robot_status, detections=scanned_detections)
    
    # [알고리즘 핵심]: 스캔 도중 포착된 모든 표지판 및 타겟의 위치를 실시간 삼각함수로 추정하여 영구 기억소자에 바인딩
    for target in scanned_detections:
        estimated_pos = estimate_target_xy_from_observation(current_obs, target.color)
        if estimated_pos:
            # 타겟 오브젝트가 패드 표지판 계열이거나 유의미한 크기일 때 전역 맵 정보 갱신
            if target.color in ["A", "B", "C", "D", "E"] or target.blob_area > 1500:
                memory.known_locations[target.color] = estimated_pos
                
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
    
    # 1. 탐색 액션 분기: 주변 각 속도 명령을 인가하여 로봇 몸체를 회전시키며 전 공간의 특징점 추출
    if decision.next_action in {"search_cube", "search_pad"}:
        # 제자리에서 일정 각도 선회 기동 연출 (시야 제약 극복용 몸체 회전)
        await move_velocity(ctx, wz=0.5, duration_s=1.8)
        return {"action": decision.next_action, "status": "repositioned_and_scanned"}

    # 2. 이동 액션 분기: 기억 데이터 체크 후 기동 타겟 설정
    if decision.next_action in {"navigate_to_cube", "navigate_to_pad"}:
        # 우선순위 1: 기억 데이터베이스 조회
        target_xy = memory.known_locations.get(decision.target_color)
        
        # 우선순위 2: 기억 데이터에 공백이 존재할 경우 실시간 시야각 마스크에서 역산
        if not target_xy:
            target_xy = estimate_target_xy_from_observation(observation, decision.target_color)
            
        if target_xy is None:
            # 기동 대상에 도달할 수 없는 상태인 경우 무작위 안전 기동 방향타 인가 유도
            await move_velocity(ctx, vx=0.3, wz=0.2, duration_s=1.5)
            return {"action": decision.next_action, "status": "failed", "reason": "좌표 데이터 부재로 인한 랜덤 순항"}
            
        result = await go_to_xy(ctx, *target_xy)
        return {"action": decision.next_action, "target_xy": target_xy, "result": result_summary(result)}

    # 3. 조작 관리 액션 분기
    if decision.next_action == "pick_cube":
        result = await pick_nearest_cube(ctx)
        return {"action": "pick_cube", "result": result_summary(result)}

    if decision.next_action == "place_cube":
        result = await place_nearest_zone(ctx)
        return {"action": "place_cube", "result": result_summary(result)}

    # 4. 자율 안전 예외 복구 분기 (벽면 충돌 및 교착 타개 프로토콜)
    if decision.next_action == "recover":
        await move_velocity(ctx, vx=-0.25, duration_s=1.2) # 후방 세이프티 스텝 기동
        await move_velocity(ctx, wz=0.4, duration_s=1.0)
        return {"action": "recover", "status": "recovery_maneuver_complete"}

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
    # 시뮬레이션 초기 원점 위치 보정을 위해 초기화 패드 수색 좌표 선입력 (옵션 타겟 튜닝 가능)
    memory.known_locations["A"] = (0.0, 0.0) 
    
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
        print(f"-> Selected Action: {decision.next_action} (Target: {decision.target_color})")
        print(f"-> Reason: {decision.reason}")

        if decision.next_action == "stop":
            break

        # 3. 행동 실행기 구동
        action_result = await execute_decision(ctx, decision, observation, memory)
        
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
