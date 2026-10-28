"""Concurrent JSON-line frontend to one resident vLLM engine."""
import argparse
import asyncio
from contextlib import redirect_stdout
import json
import sys
import time

from .common import compact, read_config


async def serve(config, checkpoint=None):
    from .backends.vllm import Engine, warm
    protocol = sys.stdout
    tasks = {}

    def send(value):
        print(compact(value), file=protocol, flush=True)

    def cancelled(rid):
        send({"request_id": rid, "ok": False, "error": "Request cancelled", "error_type": "TimeoutError"})

    def task_done(rid, task):
        if tasks.get(rid) is task:
            tasks.pop(rid, None)
        if task.cancelled():
            # Cancellation before a coroutine's first step never enters its try/finally.
            cancelled(rid)

    async def run_request(request):
        rid = request["request_id"]
        try:
            deadline = float(request["deadline"])
            queued = max(0, time.monotonic() - request.get("submitted_at", time.monotonic()))
            result = await engine.generate(rid, request, deadline)
            send({"request_id": rid, "ok": True, "worker_queue_seconds": queued, **result})
        except asyncio.CancelledError:
            cancelled(rid)
        except Exception as exc:
            send({"request_id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__})
        finally:
            tasks.pop(rid, None)

    try:
        with redirect_stdout(sys.stderr):
            engine = Engine(config, checkpoint)
            await warm(engine)
    except Exception as exc:
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    send(engine.metadata)
    try:
        # All tasks share this redirection; only send() writes to the saved protocol stream.
        with redirect_stdout(sys.stderr):
            while True:
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                request = {}
                try:
                    request = json.loads(line)
                    rid = request.get("request_id")
                    if request["operation"] == "shutdown":
                        break
                    if request["operation"] == "cancel":
                        task = tasks.get(rid)
                        if task:
                            task.cancel()
                        else:
                            cancelled(rid)  # Idempotent acknowledgement for completion/cancel races.
                        continue
                    if not isinstance(rid, str) or rid in tasks:
                        raise ValueError("Missing/duplicate request_id")
                    if request["operation"] != "generate":
                        raise ValueError("Unknown LLM operation")
                    if time.monotonic() >= float(request["deadline"]):
                        raise TimeoutError("Request expired before dispatch")
                    task = tasks[rid] = asyncio.create_task(run_request(request))
                    task.add_done_callback(lambda done, rid=rid: task_done(rid, done))
                except Exception as exc:
                    send({"request_id": request.get("request_id") if isinstance(request, dict) else None,
                          "ok": False, "error": str(exc), "error_type": type(exc).__name__})
    finally:
        pending = list(tasks.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["llm"], default="llm")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    asyncio.run(serve(read_config(args.config), args.checkpoint))


if __name__ == "__main__":
    main()
