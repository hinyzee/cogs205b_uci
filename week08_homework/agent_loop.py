#!/usr/bin/env python3

"""
Agent loop for implementing BayesFactor.

Runs a prompt -> generate -> test -> feedback cycle against the Gemini API

Setup:
    secrets/gemini.json at <repo>/secrets/gemini.json with {"api_key": "..."}

To run:
    python3 agent_loop.py 2>&1 | tee agent_loop_output.txt
"""


from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL = "gemma-4-31b-it"
#MODEL = "gemini-2.5-flash-lite"
# MODEL = "gemini-2.5-flash"

MAX_ATTEMPTS = 5
INCLUDE_TEST_FILE = True


TASK_DIR = Path(__file__).resolve().parent
TEST_DIR = TASK_DIR / "tests"
TEST_FILE = TEST_DIR / "test_bayes_factor.py"
PROMPT_FILE = TASK_DIR / "task.txt"
TARGET_FILE = TASK_DIR / "bayes_factor.py"
API_KEY_FILE = TASK_DIR.parent.parent / "secrets" / "gemini.json"

if not API_KEY_FILE.is_file():
    sys.exit(f"API key file not found: {API_KEY_FILE}")

API_KEY = json.loads(API_KEY_FILE.read_text())["GEMINI_API_KEY"]

# response schema 
# forces the model to return structured JSON
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "description": (
                "Generated files to write under the project root. Use forward "
                "slashes in relative_path. Each file's content must use real "
                "newline characters and normal indentation so it is readable."
            ),
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Basename only, e.g. bayes_factor.py.",
                    },
                    "relative_path": {
                        "type": "string",
                        "description": (
                            "POSIX-style path relative to project root, "
                            "e.g. bayes_factor.py. No leading slash, no '..'."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Full text of the file. Raw Python statements; no "
                            "markdown fences."
                        ),
                    },
                },
                "required": ["name", "relative_path", "content"],
            },
        },
        "notes": {
            "type": "string",
            "description": "Short note about design choices or trade-offs.",
        },
    },
    "required": ["files"],
}

# call gemini API
def call_gemini(prompt_text: str, attach_test_file: bool) -> dict[str, Any]:
    """POST a prompt to Gemini and return the parsed JSON response.

    Returns a dict with 'files' (list of {name, relative_path, content})
    and optionally 'notes'.
    """
    parts: list[dict[str, Any]] = [{"text": prompt_text}]

    if attach_test_file:
        blob = TEST_FILE.read_bytes()
        encoded = base64.standard_b64encode(blob).decode("ascii")
        parts.append({
            "inline_data": {
                "mime_type": "text/x-python",
                "data": encoded,
            }
        })

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": RESPONSE_SCHEMA,
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent"
    )
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            response = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        sys.exit(f"HTTP {e.code} from Gemini: {detail}")

    text = response["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)

def write_files(file_entries: list[dict[str, Any]]) -> list[Path]:
    """Write each file under TASK_DIR. Reject anything not targeting
    bayes_factor.py"""

    written: list[Path] = []

    for entry in file_entries:
        rel = entry["relative_path"]
        target = (TASK_DIR / rel).resolve()

        # Whitelist: bayes_factor.py and nothing else.
        if target != TARGET_FILE.resolve():
            print(f"  REFUSED: model tried to write {rel} (only "
                  f"bayes_factor.py is allowed)")
            continue

        target.write_text(entry["content"], encoding="utf-8")
        written.append(target)

    return written

def run_tests() -> tuple[int, str]:
    """
    Run unittest discovery; 
    
    return (returncode, combined output).
    """

    result = subprocess.run(
        ["python3", "-m", "unittest", "discover", "-s", str(TEST_DIR)],
        cwd=TASK_DIR,
        capture_output=True,
        text=True,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


# test file protection 

# set the test file to read only so the model can't modify it
TEST_FILE.chmod(0o444)

# take a snapshot of the original test file to restore later if needed
TEST_FILE_SNAPSHOT = TEST_FILE.read_text(encoding="utf-8")

# loop: prompt -> generate -> test -> feedback until tests pass

prompt_text = PROMPT_FILE.read_text()

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"\n=== Attempt {attempt} ===")

    response = call_gemini(prompt_text, attach_test_file=INCLUDE_TEST_FILE)
    files_written = write_files(response["files"])
    notes = response.get("notes", "")

    print(f"Wrote {len(files_written)} files:")
    for path in files_written:
        print(f"  + {path}")
    if notes:
        print(f"Notes: {notes}")

    # restore the test file in case the model modified it.
    TEST_FILE.chmod(0o644)
    TEST_FILE.write_text(TEST_FILE_SNAPSHOT)
    TEST_FILE.chmod(0o444)

    code, output = run_tests()
    print(f"\nOutput:\n{output}")

    # archive this attempt for the reflection.
    archive_dir = TASK_DIR / f"attempt_{attempt}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "output.txt").write_text(output)
    (archive_dir / "prompt.txt").write_text(prompt_text)
    for path in files_written:
        shutil.copy(path, archive_dir / path.name)

    if code == 0:
        print(f"\nTests passed on attempt {attempt}.")
        break

    prompt_text += (
        f"\n\nAttempt {attempt} failed.\n"
        f"```\n{output}\n```\n"
        "Fix the failures above."
    )
else:
    print(f"\nStopped after {MAX_ATTEMPTS} attempts; tests still failing.")
    sys.exit(1)