
"""Safety tests for the persistent experimental DAE bridge."""

import sys
import tempfile
import unittest

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from vibration import dae_experimental_bridge as bridge


ROOT = Path(__file__).resolve().parents[1]


FAKE_WORKER = """
import sys
import time

sys.path.insert(0, __ROOT__)

from scripts.dae_experimental_worker import serve


class FakeDAE:

    def predict_bytes(self, raw, rpm):

        if raw == b"invalid":
            raise ValueError("Invalid MAT")

        if raw == b"slow":
            time.sleep(5)

        score = 0.4 if raw == b"fault" else 0.1

        return {
            "rpm": rpm,
            "score": score,
            "threshold": 0.3,
            "predicted_anomaly": int(score > 0.3),
            "evaluation_scope": "date_based_exploratory",
        }


serve(
    FakeDAE(),
    sys.stdin.buffer,
    sys.stdout.buffer,
)
"""


class DAEExperimentalBridgeTests(unittest.TestCase):

    def setUp(self):

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.folder = Path(temp.name)

        self.worker = self.folder / "worker.py"

        self.worker.write_text(
            FAKE_WORKER.replace(
                "__ROOT__", repr(str(ROOT))
            ),
            encoding="utf-8",
        )

        self.version = 1

        patches = {
            "DAE_PYTHON": Path(sys.executable),
            "DAE_WORKER": self.worker,
            "STARTUP_TIMEOUT": 10,
            "REQUEST_TIMEOUT": 10,
            "SHUTDOWN_TIMEOUT": 2,
        }

        for name, value in patches.items():

            patcher = patch.object(
                bridge, name, value
            )

            patcher.start()
            self.addCleanup(patcher.stop)

        identity_patcher = patch.object(
            bridge,
            "artifact_identity",
            side_effect=lambda: (self.version,),
        )

        identity_patcher.start()

        self.addCleanup(identity_patcher.stop)

        self.engine = bridge.ExperimentalDAEBridge()

        self.addCleanup(self.engine.close)

    def test_initial_status(self):

        status = self.engine.status()

        self.assertFalse(status["running"])
        self.assertIsNone(status["pid"])
        self.assertEqual(status["process_starts"], 0)

    def test_process_reuse(self):

        self.engine.predict(b"normal", 900)

        first = self.engine.status()

        self.engine.predict(b"fault", 1200)
        self.engine.predict(b"normal", 1800)

        final = self.engine.status()

        self.assertEqual(first["pid"], final["pid"])
        self.assertEqual(final["process_starts"], 1)
        self.assertEqual(final["completed_requests"], 3)

    def test_normal_and_anomaly_decisions(self):

        normal = self.engine.predict(b"normal", 900)
        anomaly = self.engine.predict(b"fault", 900)

        self.assertEqual(normal["prediction"], "정상")
        self.assertEqual(anomaly["prediction"], "이상")

        self.assertEqual(normal["predicted_anomaly"], 0)
        self.assertEqual(anomaly["predicted_anomaly"], 1)

    def test_operational_policy_preserved(self):

        result = self.engine.predict(b"normal", 900)

        self.assertIs(
            result["validated_for_operation"], False
        )

        self.assertEqual(
            result["evaluation_scope"],
            "date_based_exploratory",
        )

    def test_invalid_rpm_rejected(self):

        for rpm in (0, 1500, "900", True, None):

            with self.subTest(rpm=rpm):

                with self.assertRaises(ValueError):
                    self.engine.predict(b"normal", rpm)

        self.assertEqual(
            self.engine.status()["process_starts"], 0
        )

    def test_invalid_input_rejected(self):

        inputs = (
            b"",
            "not bytes",
            None,
            bytearray(b"normal"),
            b"x" * (bridge.MAX_UPLOAD_BYTES + 1),
        )

        for raw in inputs:

            with self.subTest(input_type=type(raw)):

                with self.assertRaises(ValueError):
                    self.engine.predict(raw, 900)

        self.assertEqual(
            self.engine.status()["process_starts"], 0
        )

    def test_concurrent_requests(self):

        inputs = [
            (b"normal", 900),
            (b"fault", 1200),
            (b"normal", 1800),
            (b"fault", 900),
            (b"normal", 1200),
            (b"fault", 1800),
        ]

        def execute(item):
            return self.engine.predict(*item)

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(execute, inputs))

        for result, (raw, rpm) in zip(results, inputs):

            self.assertEqual(result["rpm"], rpm)

            self.assertEqual(
                result["predicted_anomaly"],
                int(raw == b"fault"),
            )

        status = self.engine.status()

        self.assertEqual(status["process_starts"], 1)
        self.assertEqual(status["completed_requests"], 6)

    def test_worker_crash_and_restart(self):

        self.engine.predict(b"normal", 900)

        process = self.engine._process

        process.kill()
        process.wait(timeout=5)

        result = self.engine.predict(b"normal", 900)

        self.assertEqual(result["score"], 0.1)

        self.assertEqual(
            self.engine.status()["process_starts"], 2
        )

    def test_inference_error_and_recovery(self):

        self.engine.predict(b"normal", 900)

        with self.assertRaises(RuntimeError):
            self.engine.predict(b"invalid", 900)

        self.assertFalse(
            self.engine.status()["running"]
        )

        result = self.engine.predict(b"normal", 900)

        self.assertEqual(result["score"], 0.1)

        self.assertEqual(
            self.engine.status()["process_starts"], 2
        )

    def test_request_timeout(self):

        self.engine.predict(b"normal", 900)

        with patch.object(
            bridge, "REQUEST_TIMEOUT", 0.2
        ):

            with self.assertRaises(TimeoutError):
                self.engine.predict(b"slow", 900)

        self.assertFalse(
            self.engine.status()["running"]
        )

        result = self.engine.predict(b"normal", 900)

        self.assertEqual(result["score"], 0.1)

        self.assertEqual(
            self.engine.status()["process_starts"], 2
        )

    def test_artifact_change_restarts_worker(self):

        self.engine.predict(b"normal", 900)

        self.version = 2

        self.engine.predict(b"normal", 900)

        self.assertEqual(
            self.engine.status()["process_starts"], 2
        )

    def test_explicit_shutdown(self):

        self.engine.predict(b"normal", 900)

        process = self.engine._process

        self.engine.close()

        self.assertIsNotNone(process.poll())
        self.assertFalse(self.engine.status()["running"])
        self.assertIsNone(self.engine.status()["pid"])

    def test_repeated_shutdown_is_safe(self):

        self.engine.predict(b"normal", 900)

        self.engine.close()
        self.engine.close()

        self.assertFalse(self.engine.status()["running"])

    def test_nonfinite_result_rejected(self):

        result = {
            "rpm": 900,
            "score": float("nan"),
            "threshold": 0.3,
            "predicted_anomaly": 0,
            "evaluation_scope": "date_based_exploratory",
            "validated_for_operation": False,
        }

        with self.assertRaises(ValueError):
            bridge.validate_result(result, 900)

    def test_operational_approval_rejected(self):

        result = {
            "rpm": 900,
            "score": 0.1,
            "threshold": 0.3,
            "predicted_anomaly": 0,
            "evaluation_scope": "date_based_exploratory",
            "validated_for_operation": True,
        }

        with self.assertRaises(ValueError):
            bridge.validate_result(result, 900)


if __name__ == "__main__":
    unittest.main()
