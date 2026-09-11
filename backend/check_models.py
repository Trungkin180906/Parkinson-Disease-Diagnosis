"""Check deployable model contracts without starting HTTP or loading UI code."""

import json

from backend.config import Settings
from backend.inference import ModelRegistry

if __name__ == "__main__":
    statuses = ModelRegistry(Settings().model_dir).status()
    print(json.dumps(statuses, ensure_ascii=True, indent=2))
    raise SystemExit(0 if all(value["ready"] for value in statuses.values()) else 1)
