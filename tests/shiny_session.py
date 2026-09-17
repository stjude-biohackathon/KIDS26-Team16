"""Drives the real dashboard app in-process, the way a browser would.

py-shiny ships `MockConnection` for exactly this: paired with `AppSession` it
gives a full reactive session with no web server, no browser and no extra
dependencies. Two details make it behave like a real page rather than a stub:

  * outputs are reported visible (`.clientdata_output_<id>_hidden = False`),
    otherwise Shiny suspends every one of them and nothing renders at all;
  * controls that appear inside a dynamically rendered UI are echoed back, which
    is what the browser's input bindings do. Renderers read inputs that only
    exist once some other renderer has emitted them, so a harness that skips the
    echo cannot tell a working card from a dead one.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from shiny._connection import MockConnection
from shiny.session._session import AppSession

from dashboard.interactive_dashboard import app

#: Every output the page declares, in layout.py and in nested dynamic UIs.
OUTPUT_IDS = (
    "sidebar_controls",
    "header_status_badge",
    "live_model_status_badge",
    "run_case_selector_ui",
    "run_outcome_selector_ui",
    "outcomes_card_title",
    "patient_outcomes_summary_ui",
    "executive_grade_card",
    "note_inspector_ui",
    "findings_table_ui",
    "profiling_card",
)

_SELECT = re.compile(r'<select[^>]*id="([^"]+)"(.*?)</select>', re.S)
_SELECTED_OPTION = re.compile(r'<option value="([^"]*)" selected=""')
_FIRST_OPTION = re.compile(r'<option value="([^"]*)"')
_TEXTAREA = re.compile(r'<textarea[^>]*id="([^"]+)"[^>]*>(.*?)</textarea>', re.S)
_NUMERIC = re.compile(r'<input[^>]*id="([^"]+)"[^>]*type="number"[^>]*>')
_VALUE_ATTR = re.compile(r'value="([^"]*)"')


def _reported_values(html: str) -> dict[str, Any]:
    """-> {input id: the value a browser's input bindings would report} for `html`."""
    values: dict[str, Any] = {}
    for match in _SELECT.finditer(html):
        select_id, body = match.group(1), match.group(2)
        # A multi-select reports every selected option as a list, the way the
        # browser's binding does; reporting only the first would hide a default
        # that spans several outcomes.
        if "multiple" in body.split(">", 1)[0]:
            values[select_id] = _SELECTED_OPTION.findall(body)
            continue
        chosen = _SELECTED_OPTION.search(body) or _FIRST_OPTION.search(body)
        if chosen:
            values[select_id] = chosen.group(1)
    for match in _TEXTAREA.finditer(html):
        values[match.group(1)] = match.group(2)
    for match in _NUMERIC.finditer(html):
        value = _VALUE_ATTR.search(match.group(0))
        if value and value.group(1):
            values[match.group(1)] = float(value.group(1))
        else:
            values[match.group(1)] = None
    return values


class _CapturingConnection(MockConnection):
    """A MockConnection that keeps what the server sent, instead of dropping it."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)


class FakeBrowser:
    """One dashboard session: send inputs, read back the rendered HTML."""

    def __init__(self) -> None:
        self._conn = _CapturingConnection()
        self._session = AppSession(app, "test-session", self._conn, debug=False)
        self._task: asyncio.Task | None = None
        self._read = 0
        self._echoed: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}

    async def start(self, **inputs: Any) -> None:
        self._task = asyncio.create_task(self._session._run())
        data: dict[str, Any] = dict(inputs)
        for output_id in OUTPUT_IDS:
            data[f".clientdata_output_{output_id}_hidden"] = False
        self._conn.cause_receive(json.dumps({"method": "init", "data": data}))
        await self._settle()

    async def send_inputs(self, **inputs: Any) -> None:
        self._echoed.update(inputs)
        self._conn.cause_receive(json.dumps({"method": "update", "data": inputs}))
        await self._settle()

    async def stop(self) -> None:
        self._conn.cause_disconnect()
        await asyncio.sleep(0)
        if self._task is not None:
            self._task.cancel()

    def html(self, output_id: str) -> str:
        """-> the output's rendered HTML; "" if the render was cancelled or never ran."""
        value = self.outputs.get(output_id)
        if isinstance(value, dict):
            return value.get("html") or ""
        return ""

    async def _settle(self) -> None:
        """Drain the server's messages, echoing newly rendered inputs until quiet."""
        for _ in range(40):
            await asyncio.sleep(0.02)
            new_messages = self._conn.sent[self._read:]
            self._read = len(self._conn.sent)
            echo: dict[str, Any] = {}
            for raw in new_messages:
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                for output_id, value in (message.get("values") or {}).items():
                    self.outputs[output_id] = value
                    for input_id, reported in _reported_values(self.html(output_id)).items():
                        if self._echoed.get(input_id) != reported:
                            self._echoed[input_id] = reported
                            echo[input_id] = reported
            if echo:
                self._conn.cause_receive(json.dumps({"method": "update", "data": echo}))
                continue
            if not new_messages:
                return


def _render(*, mode: str, then: dict[str, Any] | None, inputs: dict[str, Any]) -> dict[str, str]:
    async def go() -> dict[str, str]:
        browser = FakeBrowser()
        await browser.start(app_mode=mode, **inputs)
        if then:
            await browser.send_inputs(**then)
        rendered = {output_id: browser.html(output_id) for output_id in OUTPUT_IDS}
        await browser.stop()
        return rendered

    return asyncio.run(go())


def explore(run_file: str, **inputs: Any) -> dict[str, str]:
    """-> {output id: rendered HTML} for explore mode with `run_file` selected."""
    return _render(mode="explore", then={"selected_run_file": run_file}, inputs=inputs)


def live(**inputs: Any) -> dict[str, str]:
    """-> {output id: rendered HTML} for the live note evaluator, before analysis."""
    return _render(mode="live", then=None, inputs=inputs)


def analyze_live_note(note: str = "Patient admitted with a severe vaso-occlusive pain crisis.",
                      **inputs: Any) -> dict[str, str]:
    """-> {output id: rendered HTML} after clicking "Analyze & Grade Note".

    Ollama is forced offline, so the note is graded from the sidebar's typed
    feature values: the test never depends on a model being installed, and never
    fires 14 generations at whatever server happens to be running.

    The note and the click are sent together *after* the first settle, because
    the sidebar's textarea renders empty and the echo would otherwise overwrite
    a note passed at init.
    """
    import dashboard.server as server

    online = server.is_ollama_available
    server.is_ollama_available = lambda *args, **kwargs: False
    try:
        return _render(mode="live",
                       then={"live_note_text": note, "btn_analyze": 1, **inputs},
                       inputs={"outcome_present_input": True})
    finally:
        server.is_ollama_available = online


def loop_ticks_during_analysis(seconds: float = 0.3) -> int:
    """-> how many 10ms heartbeats the event loop ran while a blocking analysis
    of that length was in flight.

    The heartbeat stands in for the session's websocket ping. An analysis that
    runs on the event loop starves it: the browser gives up on the ping and the
    finished results are written to a socket that is already closed. Grading one
    note against all 14 outcomes takes minutes, so that is not a corner case.
    """
    import dashboard.server as server

    ticks = 0
    window: list[int] = []

    def blocking_analysis(**kwargs: Any) -> dict[str, dict[str, Any]]:
        window.append(ticks)
        time.sleep(seconds)
        window.append(ticks)
        return {num: {"outcome_name": f"Outcome {num}", "present": False,
                      "extracted_features": {}, "accepted_findings": [], "conflicts": {},
                      "status": "absent", "grade_result": None}
                for num in kwargs["outcomes"]}

    async def go() -> None:
        nonlocal ticks
        stop = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal ticks
            while not stop.is_set():
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        browser = FakeBrowser()
        await browser.start(app_mode="live", outcome_present_input=True)
        click = asyncio.create_task(
            browser.send_inputs(live_note_text="a note", btn_analyze=1, live_outcome=["28"]))
        # Hold the loop open until the analysis has been through its blocking
        # stretch: letting the session settle first would close the loop while the
        # worker is still asleep, and the heartbeat would stop with it.
        deadline = time.monotonic() + seconds + 5
        while len(window) < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await click
        await browser.stop()
        stop.set()
        await beat

    analysis, online = server.extract_and_grade_note, server.is_ollama_available
    server.extract_and_grade_note = blocking_analysis
    server.is_ollama_available = lambda *args, **kwargs: False
    try:
        asyncio.run(go())
    finally:
        server.extract_and_grade_note = analysis
        server.is_ollama_available = online
    return window[1] - window[0] if len(window) == 2 else -1
