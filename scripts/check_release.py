"""Scan the candidate Git tree without printing matched sensitive content.

Default reads the index (staged content); --revision reads an immutable commit.
This is a bounded pattern/allowlist check, not proof that every secret is absent.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def check(root, revision=None):
    if revision:
        files = git(root, "ls-tree", "-r", "--name-only", revision).decode().splitlines()
    else:
        files = git(root, "ls-files").decode().splitlines()
    findings = []
    for name in files:
        if any(part in name for part in (".ipynb_checkpoints/", "__pycache__/", "/phase1_snapshot/")) or Path(name).suffix.lower() in (".pkl", ".pickle", ".pyc", ".env"):
            findings.append({"path": name, "rule": "non-public-cache-or-secret-container"})
            continue
        blob = git(root, "show", f"{revision}:{name}" if revision else f":{name}")
        if Path(name).suffix.lower() == ".png":
            continue
        content = blob.decode("utf-8", errors="replace")
        if name.endswith(".ipynb"):
            try:
                notebook = json.loads(content)
                decoded = "\n".join("".join(c.get("source", [])) for c in notebook.get("cells", []))
                content += "\n" + decoded
            except ValueError:
                findings.append({"path": name, "rule": "invalid-notebook"})
        rules = {
            "literal-credential": r"(?i)(?:fred_api_key|api_key|api_secret|password|access_token)\s*=\s*['\"][a-z0-9_+/-]{20,}['\"]",
            "private-key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            "provider-token": r"(?:ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{24,})",
            "personal-absolute-path": r"(?i)[a-z]:[/\\]+u" r"sers[/\\]+|/U" r"sers/|/ho" r"me/[a-zA-Z0-9_-]+/",
            "private-report-reference": r"(?i)reports/" + "advisor|my_" + "portfolio|asesor_" + "inversiones",
        }
        for rule, pattern in rules.items():
            if re.search(pattern, content):
                findings.append({"path": name, "rule": rule})
    return {"scope": revision or "index", "files_scanned": len(files), "findings": findings, "passed": not findings, "matched_values_disclosed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(Path(__file__).resolve().parents[1], args.revision)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
