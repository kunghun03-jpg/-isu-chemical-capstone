"""Read-only experiment report lifecycle shared by the three offline CLIs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy
import sklearn

from scripts.analyze_fault_signatures import validate_output
from scripts.evaluate_fault_classifier import digest

LIMITATIONS = [
    '독립성 미확인 예비 실험: small EDA sample; Run independence is unverified.',
    'Dates/RPM may confound sensor, load, setup and measurement sequence; descriptive differences are not causal.',
    'Prior test results informed this exploratory follow-up. Reused dates are not a fresh final test set.',
    'No model adoption or web integration is authorized by these experiments alone.',
]


def default_output(kind):
    return Path.home() / 'fault-feature-experiments' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + kind)


@contextmanager
def new_report(output, protected, hashes, name, **fields):
    output = validate_output(output, protected)
    output.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'input_sha256': hashes, 'limitations': list(LIMITATIONS),
              'software': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                           'sklearn': sklearn.__version__}, **fields}
    try:
        yield output, report
        report['completed'] = True
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['inputs_unchanged'] = all(Path(p).is_file() and digest(p) == sha for p, sha in hashes.items())
        if not report['inputs_unchanged']:
            report.update(status='failed', completed=False, error='Inputs changed; generated results are invalid')
        (output / name).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def finite_ratio(numerator, denominator):
    if denominator is None or denominator <= 0 or numerator is None:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None
