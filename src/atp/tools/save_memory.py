"""save and resume with hard drive."""

import json

import atp.config as cfg

from langchain_core.messages import (
    messages_from_dict,
    messages_to_dict,
)


def save_chat(path: str, state: cfg.BaseState) -> None:
    """Save chat memory to hard drive."""
    serialized_messages = messages_to_dict(state["messages"])

    data = {
        "messages": serialized_messages,
    }

    # 3. 写入文件
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def resume_chat(path: str) -> cfg.BaseState:
    """Resume chat memory from hard drive."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["messages"] = messages_from_dict(data["messages"])

        return data
    except FileNotFoundError as e:
        raise e


if __name__ == "__main__":
    from pprint import pprint

    pprint(resume_chat("history.json"))
