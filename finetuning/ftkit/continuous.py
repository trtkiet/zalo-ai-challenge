"""Bounded rolling admission and correlated, nonblocking model transports."""
from collections import deque
import queue
import threading
import time

from .common import compact
from .submission import Worker, sample_steps


class AsyncWorker(Worker):
    """The stdout reader is shared; request completion never blocks admission/polling."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_ids = set()
        self.cancelled_at = {}
        self.sequence = 0
        self.writer_process = None
        self.outbound = None

    def start(self, deadline):
        if self.pending_ids and (self.process is None or self.process.poll() is not None):
            raise RuntimeError("Worker exited with outstanding requests; poll their failures before restarting")
        super().start(deadline)
        if self.writer_process is self.process:
            return
        process, messages = self.process, self.messages
        outbound = self.outbound = queue.Queue(maxsize=1024)
        self.writer_process = process

        def writer():
            while True:
                value = outbound.get()
                if value is None:
                    return
                try:
                    process.stdin.write(value + "\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    messages.put(None)
                    return
        threading.Thread(target=writer, daemon=True).start()

    def send(self, value):
        try:
            self.outbound.put_nowait(compact(value))
        except queue.Full as exc:
            raise RuntimeError("Worker input queue full; lower active sample concurrency") from exc

    def submit(self, operation, payload, deadline):
        self.start(deadline)
        self.sequence += 1
        rid = f"{self.command[self.command.index('--kind') + 1]}-{self.sequence}"
        request = {**payload, "operation": operation, "request_id": rid, "deadline": deadline,
                   "submitted_at": time.monotonic(),
                   "seconds": max(0.001, deadline - time.monotonic())}
        self.send(request)
        self.pending_ids.add(rid)
        return rid

    def poll(self):
        completed = []
        while True:
            try:
                response = self.messages.get_nowait()
            except queue.Empty:
                break
            if response is None:
                completed.extend({"request_id": rid, "ok": False, "error": f"Worker exited; see {self.log_path}"}
                                 for rid in self.pending_ids)
                self.pending_ids.clear()
                self.close()
                break
            rid = response.get("request_id")
            self.cancelled_at.pop(rid, None)
            if rid in self.pending_ids:
                self.pending_ids.remove(rid)
                completed.append(response)
        dead = self.process is not None and self.process.poll() is not None
        if dead or any(time.monotonic() - when > 2 for when in self.cancelled_at.values()):
            completed.extend({"request_id": rid, "ok": False, "error": "Worker did not acknowledge cancellation; process group reaped"}
                             for rid in self.pending_ids)
            self.close()
        return completed

    def cancel(self, rid):
        self.pending_ids.discard(rid)
        self.cancelled_at[rid] = time.monotonic()
        if self.process is not None and self.process.poll() is None:
            try:
                self.send({"operation": "cancel", "request_id": rid})
            except RuntimeError:
                pass

    def close(self):
        if self.outbound is not None:
            try:
                self.outbound.put_nowait(None)
            except queue.Full:
                pass
        super().close()
        self.writer_process = None
        self.outbound = None
        self.pending_ids.clear()
        self.cancelled_at.clear()


def continuous_samples(items, tools, package, workers, root, split, *, capacity=1, on_admit=None, should_stop=None, **settings):
    """Yield completed samples immediately. The caller durably journals each before resuming.

    ASR batches only requests already ready at dispatch. LLM requests each have a
    unique ID and deadline; a single timeout never kills a shared engine.
    """
    if capacity < 1:
        raise ValueError("Active sample capacity must be positive")
    active, pending, waiting = {}, {}, {}
    completed = deque()
    source = iter(items)
    exhausted = stopping = False
    asr_busy = None
    asr_orphan_since = None

    def advance(sid, response=None, error=None):
        try:
            gen = active[sid]["steps"]
            pending[sid] = gen.throw(error) if error is not None else gen.send(response)
            active[sid]["queued_at"] = time.monotonic()
        except StopIteration as done:
            pending.pop(sid, None)
            waiting.pop(sid, None)
            result, trace = done.value
            trace["admission_index"] = active[sid]["index"]
            trace["admitted_at_monotonic"] = active[sid]["started"]
            completed.append((result, trace))

    admission_index = 0
    try:
        while active or not exhausted or completed:
            while completed:
                result, trace = completed.popleft()
                yield result, trace
                active.pop(result["id"], None)
            if should_stop and should_stop():
                raise KeyboardInterrupt
            while not stopping and not exhausted and len(active) < capacity:
                try:
                    item = next(source)
                except StopIteration:
                    exhausted = True
                    break
                sid = item["id"]
                if on_admit:
                    on_admit(sid)
                started = time.monotonic()
                env = package.ToolEnv(split, sid, root=root)
                active[sid] = {"index": admission_index, "started": started,
                               "steps": sample_steps(item, tools, env, root, split, started=started, **settings)}
                admission_index += 1
                advance(sid)
            if not active and exhausted:
                break
            # Model results already available before their deadlines win over later work.
            for kind in ("asr", "llm"):
                for response in workers[kind].poll():
                    rid = response.get("request_id")
                    if kind == "asr":
                        if rid != asr_busy:
                            continue
                        indices = [sid for sid, value in waiting.items() if value == ("asr", rid)]
                        all_indices = asr_indices
                        asr_busy, asr_orphan_since = None, None
                        values = response.get("results")
                        if response.get("ok") and (not isinstance(values, list) or len(values) != len(all_indices)):
                            response = {"ok": False, "error": "ASR returned mismatched batch results"}
                        responses = {sid: values[i] for i, sid in enumerate(all_indices)} if response.get("ok") else {}
                    else:
                        indices = [sid for sid, value in waiting.items() if value == ("llm", rid)]
                        responses = {sid: response for sid in indices}
                    for sid in indices:
                        waiting.pop(sid, None)
                        value = responses.get(sid, response)
                        value["scheduler_wait_seconds"] = active[sid].get("scheduler_wait_seconds", 0)
                        if time.monotonic() >= pending[sid][3]:
                            advance(sid, error=TimeoutError("Sample deadline exceeded"))
                        elif value.get("ok"):
                            advance(sid, response=value)
                        else:
                            exc = TimeoutError if value.get("error_type") == "TimeoutError" else RuntimeError
                            advance(sid, error=exc(value.get("error", "Model request failed")))
            for sid in list(pending):
                if time.monotonic() >= pending[sid][3]:
                    transport = waiting.pop(sid, None)
                    if transport and transport[0] == "llm":
                        workers["llm"].cancel(transport[1])
                    advance(sid, error=TimeoutError("Sample deadline exceeded"))
            # ASR cannot be interrupted within a synchronous CUDA batch. Once no live
            # sample owns that work, give it a short grace period then reap its group.
            if asr_busy and not any(value == ("asr", asr_busy) for value in waiting.values()):
                asr_orphan_since = time.monotonic() if asr_orphan_since is None else asr_orphan_since
                if time.monotonic() - asr_orphan_since > 2:
                    workers["asr"].close()
                    asr_busy, asr_orphan_since = None, None
            if not asr_busy:
                asr_indices = [sid for sid, req in pending.items() if req[0] == "asr" and sid not in waiting]
                if asr_indices:
                    try:
                        requests = [{**pending[sid][2], "deadline": pending[sid][3]} for sid in asr_indices]
                        asr_busy = workers["asr"].submit("transcribe_batch", {"requests": requests},
                                                         max(pending[sid][3] for sid in asr_indices))
                        for sid in asr_indices:
                            active[sid]["scheduler_wait_seconds"] = time.monotonic() - active[sid]["queued_at"]
                            waiting[sid] = ("asr", asr_busy)
                    except Exception as exc:
                        asr_busy = None
                        for sid in asr_indices:
                            advance(sid, error=exc)
            for sid in list(pending):
                kind, operation, payload, deadline = pending[sid]
                if kind == "llm" and sid not in waiting:
                    try:
                        active[sid]["scheduler_wait_seconds"] = time.monotonic() - active[sid]["queued_at"]
                        rid = workers["llm"].submit(operation, payload, deadline)
                        waiting[sid] = ("llm", rid)
                    except Exception as exc:
                        advance(sid, error=exc)
            if completed:
                continue
            if pending:
                time.sleep(0.002)
    except KeyboardInterrupt:
        stopping = True
        for sid, (kind, rid) in list(waiting.items()):
            if kind == "llm":
                workers[kind].cancel(rid)
        for sid in list(pending):
            advance(sid, error=KeyboardInterrupt())
        while completed:
            yield completed.popleft()

