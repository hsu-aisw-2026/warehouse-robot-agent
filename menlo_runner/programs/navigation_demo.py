# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.
# 스타터 코드: 프로젝트 제출을 위해 이 공용 파일을 직접 수정하지 마세요.
# 프로젝트 코드는 프로젝트 노트북 또는 menlo_runner/programs/의 새 파일에 작성하세요.

from __future__ import annotations

from menlo_runner.perception import perceive
from menlo_runner.navigation import my_go_to_global, my_go_to_visual


async def run(ctx) -> None:
    print("Part A: custom global-state navigation to pad_C")
    reached = await my_go_to_global(ctx, "pad_C", tolerance_m=0.8, max_iters=3)
    print(f"Global navigation result: {reached}")

    print("\nPart B: vision-only navigation to the first visible cube color")
    obs = await perceive(ctx)
    if not obs:
        print("No visible cube colors. Move/reset the robot near the conveyor and try again.")
        return
    target_color = next(iter(obs))
    reached = await my_go_to_visual(ctx, target_color)
    print(f"Vision navigation result for {target_color}: {reached}")

