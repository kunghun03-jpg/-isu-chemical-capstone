
"""Tests for the persistent experimental DAE worker."""

import io
import json
import struct
import unittest

from scripts.dae_experimental_worker import (
    serve,
    read_exact,
    MAX_HEADER_BYTES,
    MAX_UPLOAD_BYTES,
)


class FakeDAE:

    def __init__(self):
        self.calls = 0

    def predict_bytes(self, raw, rpm):

        self.calls += 1

        if raw == b"invalid":
            raise ValueError("Invalid MAT")

        score = 0.4 if raw == b"fault" else 0.1

        return {
            "rpm": rpm,
            "score": score,
            "threshold": 0.3,
            "predicted_anomaly": int(score > 0.3),
            "evaluation_scope": "date_based_exploratory",
        }


def frame(header, raw=b""):

    encoded = json.dumps(header).encode("utf-8")

    return (
        struct.pack("!I", len(encoded))
        + encoded
        + raw
    )


def request(request_id, rpm=900, raw=b"normal"):

    return frame({
        "command": "predict",
        "request_id": request_id,
        "rpm": rpm,
        "size": len(raw),
    }, raw)


def shutdown(request_id):

    return frame({
        "command": "shutdown",
        "request_id": request_id,
    })


class DAEExperimentalWorkerTests(unittest.TestCase):

    def run_worker(self, payload, engine=None):

        if engine is None:
            engine = FakeDAE()

        output = io.BytesIO()

        serve(
            engine,
            io.BytesIO(payload),
            output,
        )

        responses = [
            json.loads(line)
            for line in output.getvalue().splitlines()
        ]

        return responses, engine

    def test_ready_response(self):

        rows, _ = self.run_worker(b"")

        self.assertEqual(rows[0]["status"], "ready")
        self.assertEqual(rows[0]["protocol_version"], 1)
        self.assertIs(
            rows[0]["validated_for_operation"], False
        )

    def test_multiple_rpms_reuse_engine(self):

        rows, engine = self.run_worker(
            request(1, 900)
            + request(2, 1200, b"fault")
            + request(3, 1800)
        )

        self.assertEqual(engine.calls, 3)

        self.assertEqual(
            [r["result"]["rpm"] for r in rows[1:]],
            [900, 1200, 1800],
        )

        self.assertEqual(
            [
                r["result"]["predicted_anomaly"]
                for r in rows[1:]
            ],
            [0, 1, 0],
        )

    def test_operational_policy_preserved(self):

        rows, _ = self.run_worker(request(1))

        result = rows[1]["result"]

        self.assertIs(
            result["validated_for_operation"], False
        )

        self.assertEqual(
            result["evaluation_scope"],
            "date_based_exploratory",
        )

    def test_request_id_preserved(self):

        rows, _ = self.run_worker(
            request(17) + request(35)
        )

        self.assertEqual(rows[1]["request_id"], 17)
        self.assertEqual(rows[2]["request_id"], 35)

    def test_invalid_rpm_and_recovery(self):

        rows, engine = self.run_worker(
            request(1, 1500) + request(2, 900)
        )

        self.assertEqual(rows[1]["status"], "error")
        self.assertEqual(rows[2]["status"], "ok")
        self.assertEqual(engine.calls, 1)

    def test_invalid_mat_and_recovery(self):

        rows, engine = self.run_worker(
            request(1, raw=b"invalid")
            + request(2)
        )

        self.assertEqual(rows[1]["status"], "error")
        self.assertEqual(rows[2]["status"], "ok")
        self.assertEqual(engine.calls, 2)

    def test_graceful_shutdown(self):

        rows, engine = self.run_worker(
            request(1)
            + shutdown(2)
            + request(3)
        )

        self.assertEqual(rows[-1]["status"], "shutdown")
        self.assertEqual(rows[-1]["request_id"], 2)
        self.assertEqual(engine.calls, 1)

    def test_oversized_header_rejected(self):

        payload = struct.pack(
            "!I", MAX_HEADER_BYTES + 1
        )

        rows, engine = self.run_worker(payload)

        self.assertEqual(
            rows[-1]["status"], "protocol_error"
        )

        self.assertEqual(engine.calls, 0)

    def test_oversized_payload_rejected(self):

        payload = frame({
            "command": "predict",
            "request_id": 1,
            "rpm": 900,
            "size": MAX_UPLOAD_BYTES + 1,
        })

        rows, engine = self.run_worker(payload)

        self.assertEqual(
            rows[-1]["status"], "protocol_error"
        )

        self.assertEqual(engine.calls, 0)

    def test_incomplete_payload_rejected(self):

        payload = frame({
            "command": "predict",
            "request_id": 1,
            "rpm": 900,
            "size": 100,
        }, b"short")

        rows, engine = self.run_worker(payload)

        self.assertEqual(
            rows[-1]["status"], "protocol_error"
        )

        self.assertEqual(engine.calls, 0)

    def test_invalid_json_rejected(self):

        raw = b"not valid json"

        payload = struct.pack("!I", len(raw)) + raw

        rows, _ = self.run_worker(payload)

        self.assertEqual(
            rows[-1]["status"], "protocol_error"
        )

    def test_unknown_command_rejected(self):

        rows, _ = self.run_worker(
            frame({
                "command": "unknown",
                "request_id": 1,
            })
        )

        self.assertEqual(
            rows[-1]["status"], "protocol_error"
        )

    def test_read_exact_rejects_truncated_stream(self):

        with self.assertRaises(EOFError):

            read_exact(io.BytesIO(b"abc"), 5)

    def test_nonfinite_score_rejected(self):

        class InvalidDAE(FakeDAE):

            def predict_bytes(self, raw, rpm):

                result = super().predict_bytes(raw, rpm)
                result["score"] = float("nan")

                return result

        rows, _ = self.run_worker(
            request(1),
            InvalidDAE(),
        )

        self.assertEqual(rows[-1]["status"], "error")

    def test_inconsistent_decision_rejected(self):

        class InvalidDAE(FakeDAE):

            def predict_bytes(self, raw, rpm):

                result = super().predict_bytes(raw, rpm)
                result["predicted_anomaly"] = 1

                return result

        rows, _ = self.run_worker(
            request(1),
            InvalidDAE(),
        )

        self.assertEqual(rows[-1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
