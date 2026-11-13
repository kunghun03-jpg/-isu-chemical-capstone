"""Create an isolated WSL web environment without changing ~/capstone/.venv."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MARKER = ".capstone-web-env.json"
PINS = ("numpy==2.4.6", "torch==2.14.1+cpu", "scikit-learn==1.9.0",
        "scipy==1.18.1", "streamlit==1.67.0")
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def validate_target(target, project=ROOT, home=None):
    home = Path.home() if home is None else Path(home)
    original = Path(target).expanduser()
    if original.is_symlink():
        raise ValueError("가상환경 경로는 심볼릭 링크일 수 없습니다.")
    target = original.resolve()
    for protected in ((home / "capstone/.venv").resolve(), (Path(project) / ".venv").resolve()):
        if target.is_relative_to(protected) or protected.is_relative_to(target):
            raise ValueError("기존 .venv 또는 그 상위/하위 경로는 변경할 수 없습니다.")
    # Keep installed packages, setup reports and markers out of every checkout.
    if target.is_relative_to(Path(project).resolve()) or any(
        (p / ".git/HEAD").is_file() or (p / ".git").is_file() for p in (target, *target.parents)
    ):
        raise ValueError("웹 가상환경은 Git 체크아웃 밖에 생성하세요.")
    if target.exists():
        marker = target / MARKER
        if not marker.is_file() or not (target / "pyvenv.cfg").is_file():
            raise ValueError("이 도구가 생성하지 않은 기존 경로는 변경하지 않습니다.")
        metadata = json.loads(marker.read_text())
        if metadata.get("project") != str(Path(project).resolve()):
            raise ValueError("다른 프로젝트의 웹 가상환경은 변경하지 않습니다.")
    return target


def run(command, *, environment=None, capture=False):
    return subprocess.run(command, cwd=ROOT, env=environment, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None)


def python_info(executable):
    result = run([str(executable), "-B", "-c",
                  "import json,sys; print(json.dumps(list(sys.version_info[:3])))"], capture=True)
    return json.loads(result.stdout)


def setup(target, interpreter, bundle_root):
    target = validate_target(target)
    version = python_info(interpreter)
    if version[:2] not in ([3, 14], [3, 12]):
        raise ValueError(f"Python {version}: Python 3.14.4 또는 3.12.x를 지정하세요.")
    print(f"대상 Python: {'.'.join(map(str, version))}; 기존 .venv 변경 없음", flush=True)
    if target.exists():
        existing = python_info(target / "bin/python")
        if existing[:2] != version[:2]:
            raise ValueError("기존 웹 환경의 Python 계열이 다릅니다. --env로 새 경로를 지정하세요.")
    else:
        run([str(interpreter), "-B", "-m", "venv", str(target)])
        (target / MARKER).write_text(json.dumps({"project": str(ROOT), "python": version}) + "\n")
    python = str(target / "bin/python")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PIP_REQUIRE_VIRTUALENV="true",
                       PIP_CACHE_DIR=str(target / "pip-cache"), PIP_CONFIG_FILE=os.devnull)
    # A pre-existing pip target/prefix/user setting must not redirect writes to
    # the user's old environment. Keep proxy and CA environment bindings intact.
    for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_USER", "PIP_PYTHON", "PIP_TRUSTED_HOST"):
        environment.pop(key, None)
    # Honor inherited proxy and CA settings. Never disable TLS verification or
    # silently fall back to building wheels / replacing the user's environment.
    base = [python, "-B", "-m", "pip", "install", "--only-binary=:all:"]
    cpu = ["torch==2.14.1+cpu", "--index-url", CPU_INDEX]
    packages = ["-r", str(ROOT / "requirements.txt"), *PINS]
    run(base + ["--dry-run", *cpu], environment=environment)
    run(base + cpu, environment=environment)
    # Check the complete CPython-specific binary dependency solution before
    # installation; torch is already present from the official CPU index.
    run(base + ["--dry-run", *packages], environment=environment)
    run(base + packages, environment=environment)
    run([python, "-B", "-m", "pip", "check"], environment=environment)
    result = subprocess.run(
        [python, "-B", "-m", "scripts.verify_web_env", "--bundle-root", str(bundle_root)],
        cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    (target / "bundle-loading-report.json").write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        (target / "bundle-loading-stderr.log").write_text(result.stderr, encoding="utf-8")
    print(result.stdout, end="")
    if result.returncode:
        print(f"환경 설치 후 번들 로딩 미확인/실패. 보고서: {target / 'bundle-loading-report.json'}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return 2
    print("세 RPM의 AE·OCSVM·IF 번들 로딩 확인 완료. 실제 MAT 추론은 별도 확인이 필요합니다.")
    print(f"cd {shlex.quote(str(ROOT))}\n{shlex.quote(python)} -m streamlit run streamlit_app.py")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=Path.home() / "capstone-web-venv")
    parser.add_argument("--python", dest="interpreter", help="예: python3.14 또는 python3.12")
    parser.add_argument("--bundle-root", type=Path,
                        default=ROOT / "outputs/inference_bundles/date_prelim_20261106")
    args = parser.parse_args()
    interpreter = args.interpreter or shutil.which("python3.14")
    if not interpreter and sys.version_info[:2] == (3, 14):
        interpreter = sys.executable
    if not interpreter:
        parser.error("Python 3.14를 찾지 못했습니다. --python python3.12 --env ~/capstone-web-py312 대안을 사용하세요.")
    try:
        return setup(args.env, interpreter, args.bundle_root.expanduser().resolve())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"설정 실패: {error}\nPython 3.12 대안: --python python3.12 --env ~/capstone-web-py312\n"
              "실패가 번들 누락/버전/해시 문제라면 Python 변경만으로 해결되지 않습니다.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
