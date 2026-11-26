
"""Persistent worker for experimental DAE inference.

Keep the model in memory without changing operational DAE policy.
"""

import json
import math
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_HEADER_BYTES = 4096

SUPPORTED_RPMS = (900, 1200, 1800)


def read_exact(stream, size):
    """Read exactly the requested number of bytes."""

    if size < 0:
        raise ValueError("Invalid read size")

    chunks = []
    remaining = size

    while remaining:
        chunk = stream.read(remaining)

        if not chunk:
            raise EOFError("Incomplete request")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def send_response(stream, response):
    """Write a single JSON response."""

    payload = json.dumps(
        response,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    stream.write(payload + b"\n")
    stream.flush()


def serve(engine, input_stream, output_stream):
    """Process multiple requests using one loaded model."""

    send_response(
        output_stream,
        {
            "status": "ready",
            "protocol_version": 1,
            "validated_for_operation": False,
        },
    )

    while True:

        prefix = input_stream.read(4)

        if not prefix:
            return

        try:
            if len(prefix) < 4:
                prefix += read_exact(
                    input_stream,
                    4 - len(prefix),
                )

            header_size = struct.unpack(
                "!I", prefix
            )[0]

            if not 0 < header_size <= MAX_HEADER_BYTES:
                raise ValueError("Invalid header size")

            header = json.loads(
                read_exact(
                    input_stream,
                    header_size,
                ).decode("utf-8")
            )

            if not isinstance(header, dict):
                raise ValueError("Invalid request header")

            request_id = header.get("request_id")
            command = header.get("command")

            if type(request_id) is not int or request_id < 0:
                raise ValueError("Invalid request ID")

            if command == "shutdown":

                send_response(
                    output_stream,
                    {
                        "status": "shutdown",
                        "request_id": request_id,
                    },
                )

                return

            if command != "predict":
                raise ValueError("Unknown command")

            rpm = header.get("rpm")
            size = header.get("size")

            if (
                type(size) is not int
                or not 0 < size <= MAX_UPLOAD_BYTES
            ):
                raise ValueError("Invalid MAT size")

            # Consume the complete payload before
            # processing or rejecting this request.
            raw = read_exact(input_stream, size)

        except Exception:

            send_response(
                output_stream,
                {
                    "status": "protocol_error",
                    "error": "Invalid DAE worker request",
                },
            )

            # Stream alignment is no longer guaranteed.
            return

        try:

            if (
                type(rpm) is not int
                or rpm not in SUPPORTED_RPMS
            ):
                raise ValueError("Unsupported RPM")

            result = engine.predict_bytes(raw, rpm)

            score = float(result["score"])
            threshold = float(result["threshold"])
            decision = result["predicted_anomaly"]

            if not (
                math.isfinite(score)
                and math.isfinite(threshold)
                and score >= 0
                and threshold > 0
            ):
                raise ValueError("Invalid DAE score")

            if (
                type(decision) is not int
                or decision not in (0, 1)
                or decision != int(score > threshold)
            ):
                raise ValueError("Invalid DAE decision")

            if result["rpm"] != rpm:
                raise ValueError("RPM mismatch")

            if (
                result.get("evaluation_scope")
                != "date_based_exploratory"
            ):
                raise ValueError("Invalid evaluation scope")

            send_response(
                output_stream,
                {
                    "status": "ok",
                    "request_id": request_id,
                    "result": {
                        "rpm": rpm,
                        "score": score,
                        "threshold": threshold,
                        "predicted_anomaly": decision,
                        "evaluation_scope":
                            "date_based_exploratory",
                        "validated_for_operation": False,
                    },
                },
            )

        except Exception as error:

            print(
                f"DAE inference failed: {type(error).__name__}",
                file=sys.stderr,
                flush=True,
            )

            send_response(
                output_stream,
                {
                    "status": "error",
                    "request_id": request_id,
                    "error": "DAE experimental inference failed",
                },
            )


def main():

    from scripts.dae_experimental_inference import (
        ExperimentalDAE,
    )

    engine = ExperimentalDAE()

    serve(
        engine,
        sys.stdin.buffer,
        sys.stdout.buffer,
    )


if __name__ == "__main__":
    main()
