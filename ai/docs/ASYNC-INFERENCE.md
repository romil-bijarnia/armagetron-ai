# Blacklight local inference

The local in-game policy runs on one CPU worker. The game thread still builds observations
at `NEURAL_DECISION_INTERVAL` (normally 0.05 seconds) and applies controls; it never waits for
`Net::Decide` during ordinary play. External socket control and native teacher recording keep
their existing synchronous protocol.

Each job owns copied maps, scalars, its legal mask, sampling setting and identifiers. The worker
cannot read game objects, graphics state or mutable configuration. Each configured player slot
has at most one pending observation and one completed result, in addition to the one inference
in progress. New observations replace pending ones, and slots are processed round-robin.

Results carry weight generation, round, player, slot, life generation and decision sequence.
The game thread rejects results that no longer match the latest request, belong to a dead or
replaced cycle, have already been applied, or are older than one observation interval in game
time. It recomputes legality before applying the action. Non-owning object observers distinguish
a new cycle even if an old memory address is reused.

The action arrives after inference plus the next main/world tick, rather than freezing the
render loop while inference finishes. Until a valid result arrives, the cycle continues its
current heading and brake state. An overloaded worker drops obsolete work instead of building
a queue of delayed actions. This changes control latency and should be measured in playing
evaluations as well as frame-time tests.

Memory and the sampling RNG are worker-owned and separate for each life. Only processed
observations advance them. New lives, new weights and skipped decision sequences reset Memory;
skipped observations are not silently treated as consecutive. This also defines the reset
contract for any later recurrent state. No recurrent layer is introduced by this change.

Model loading joins the old worker before modifying weights. Application shutdown joins the
worker before graphics and static inference buffers are destroyed. The normal frame path only
holds the queue mutex briefly for submission/result transfer; inference runs outside it.

With `NEURAL_DEBUG` enabled, the periodic `neural worker` line reports mean inference time,
queued-plus-inference time, end-to-end application latency, applied/stale/replaced counts,
life resets, gap resets and skipped decisions. GPU rendering is not part of those worker timings.

Run the deterministic CPU-only lifecycle checks from `ai`:

```sh
.venv/bin/python -m pytest tests/test_async_brain.py -q
```

The checks deliberately block inference while submitting/polling, verify latest-only queueing,
per-slot memory and reset ordering, reject changed identities/legality/age, and confirm that
reload/shutdown waits for the worker before replacing its resources.

For local frame-time measurements, launch the client with `--hidden-window --local-game`
and isolated user configuration, data and var directories. This enters the normal single-player
game with its saved bot settings. A network spectator preview does not exercise local bot
inference or the local simulation. `--local-game` requires `--hidden-window` and cannot be
combined with `--connect`.

Set `FRAME_TELEMETRY 1` to append aggregate frame timings every five seconds to
`frame-times.log` in the game's var directory. The default is off. Logging also works in
ordinary visible play and does no per-frame file I/O. Reports separate active-game frames
from menus and include percentiles, slow-frame counts, the drawable size, actual SDL
fullscreen and focus flags, display refresh, swap interval and antialiasing samples. Read the
actual fullscreen flag: a saved fullscreen preference does not make a hidden test fullscreen.

On macOS the normal log is
`~/Library/Application Support/Armagetron Advanced/var/frame-times.log`. Keep hidden-window
results labelled as such; they do not establish fullscreen presentation performance.
