from minisweagent.exceptions import Submitted


class _FakeMiniAgent:
    def __init__(self, messages):
        self.messages = messages
        self.model = _FakeMiniModel()
        self.env = _FakeMiniEnv()

    def add_messages(self, message):
        self.messages.append(message)


class _FakeMiniEnv:
    def __init__(self):
        self.executed_commands: list[str] = []

    def execute(self, action: dict):
        self.executed_commands.append(action.get("command", ""))
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in action.get("command", ""):
            raise Submitted({"role": "exit", "content": "diff", "extra": {"exit_status": "Submitted"}})
        return {"output": "", "returncode": 0}


class _FakeMiniModel:
    @staticmethod
    def format_message(role, content, extra=None):
        return {"role": role, "content": content, "extra": extra or {}}
