# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.
# 스타터 코드: 프로젝트 제출을 위해 이 공용 파일을 직접 수정하지 마세요.
# 프로젝트 코드는 프로젝트 노트북 또는 menlo_runner/programs/의 새 파일에 작성하세요.

from __future__ import annotations

from menlo_runner.basics import print_position, screenshot
from menlo_runner.scene import get_scene_text


async def run(ctx) -> None:
    print("Robot status:")
    await print_position(ctx, "CURRENT")

    print("\nScene summary:")
    print(await get_scene_text(ctx))

    await screenshot(ctx, "Saved current POV:", "outputs/basics-demo-pov.jpg")

    print("\nSmall SDK movement demo:")
    await ctx.invoke("set_velocity", {"vx": 0.4, "vy": 0.0, "wz": 0.0, "duration_s": 1.0})
    await print_position(ctx, "AFTER")

