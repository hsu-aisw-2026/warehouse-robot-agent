# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.
# 스타터 코드: 프로젝트 제출을 위해 이 공용 파일을 직접 수정하지 마세요.
# 프로젝트 코드는 프로젝트 노트북 또는 menlo_runner/programs/의 새 파일에 작성하세요.

from __future__ import annotations

from menlo_runner.agents import WorkshopAgent


TASK = (
    "Use get_scene_summary to find a visible cube, go to it, pick it up, "
    "check what you are holding, and place it on the correct pad. "
    "Call done after one successful delivery or if you cannot continue."
)


async def run(ctx) -> None:
    agent = WorkshopAgent(ctx, tokamak_api_key=ctx.config.tokamak_api_key)
    _messages, tool_log = await agent.run(TASK, max_turns=12)
    print("\nTool log:")
    for entry in tool_log:
        print(f"  turn {entry['turn']}: {entry['tool']} -> {entry['result'][:80]}")

