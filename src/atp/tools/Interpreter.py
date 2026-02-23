import asyncio

import pantograph as ptg

server = ptg.Server(project_path="./ATP")


server = asyncio.run(server.create())
state0 = asyncio.run(server.goal_start_async("forall (p q: Prop), Or p q -> Or q p"))

print(state0)
