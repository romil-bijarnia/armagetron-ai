# Blacklight — Armagetron AI

Blacklight is our AI for Armagetron Advanced, with a policy that can run inside the game without Python.

`./blacklight train` continues the existing model with reinforcement learning against scripted bots in the native game. Blacklight chooses its own moves and learns from complete match outcomes. The separate `arma-teach` simulator trainer remains available for explicitly requested self-play experiments. The implemented combat architecture is documented in [docs/V4.md](docs/V4.md).

## What the network sees

v4 reads four 64×64 maps: Nearby at 2 m per cell, Close at 1 m, the whole arena, and engine-computed territory/distance. Current and previous frames are stacked. Nearby and Close use separate progressive towers that process geometry at 32×32 before further reduction. The arena tower conditions both local branches. Of the 108 engine scalars, 72 own/arena numbers enter their own branch; three 12-number enemy records enter a shared, masked opponent encoder and context-dependent attention pool. Enemy records include braking reserve, rubber usage, and separate left/right turn waits.

The policy has six joint controls: straight, left, right, brake, left + brake and right + brake. Illegal turns are masked. V4 has **8,155,373 parameters**, persistent 256-unit memory, a bounded value head, and auxiliary predictions for territory, danger, a credited kill while surviving the next two seconds, and future ownership. Memory also receives the previous executed control and elapsed time. [Full architecture and verification](docs/V4.md).

## Setting up

Install the Python environment once, from this folder. If the repository sits in an iCloud-synced folder (Desktop or Documents), keep the environment outside it: iCloud marks files it manages as hidden, and Python 3.13 ignores hidden `.pth` files, which breaks the `arma-*` commands.

```bash
uv venv --python 3.13 ~/.venvs/armagetron-ai && ln -s ~/.venvs/armagetron-ai .venv && uv sync
```

## Native bot training

From this directory:

```bash
./blacklight train          # bot-only RL; two engines, background resource budget
./blacklight train full     # bot-only RL; eight engines, uncapped work
./blacklight watch          # live compact progress; Ctrl-C only closes this view
./blacklight status
./blacklight stop           # finish the current learning update, save and stop
```

The native monitor is a compact status block. Its bar describes the current batch,
the win rate names its recent-game window, and session games, elapsed time,
throughput and checkpoint age are separate. Continuous training has no overall
completion percentage. Restarting the monitor reloads display changes without
stopping the trainer.

The six custom profiles below are the opponents. Each match has exactly one
neural-controlled player; all opponents use the native scripted controller.
Duels and four-player matches vary across small, medium and full-size arenas.
No self-play opponent or teacher action label enters this mode.

The learner samples its own legal controls, records their probabilities and
recurrent histories, and trains the policy and outcome-value prediction from
completed lives. Wins return +1, deaths -1 and surviving timeouts 0, discounted
back over the life. Auxiliary prediction heads retain their existing weights;
this first native mode does not invent auxiliary labels from unavailable data.
Engines finish their collection batch before the model changes, and partial lives
are discarded when stopping. Initial time-zero controls ignored by the engine are
excluded from learning.

Training resumes `runs/blacklight/latest.pt`. Before the first native session,
`snapshots/before-native-bots-uXXXXXXXX.pt` preserves its starting checkpoint.
Saves occur every 60 seconds when learning has advanced and on a clean stop;
failed updates do not overwrite the last saved checkpoint. Native sessions resume
their optimizer state, and changing training objectives resets optimizer moments
while retaining the learned network. The same exclusive writer lock protects all
training methods.

Background mode uses low process priority and a 35% work/rest target. Full mode
keeps the Mac awake. `--engines`, `--lr` and the other options in
`./blacklight train --help` override the presets. Reported match outcomes are
training results; independent real-engine evaluation still establishes strength.

## Simulator training (optional)

The older simulator workflow uses self-play, Hunter and an exploiter. It is
separate from the native-bot command above and resumes the same checkpoint:

```bash
./trainctl background      # simulator self-play, background resource budget
./trainctl full            # simulator self-play, whole machine
```

`./blacklight` alone lists the rest: `stop`, `status`, `watch`, `dash`, `log`,
`ladder`, `show`, `brain`. The two modes are `background` and `full`; the
`uv run arma-background` and `uv run arma-full` commands are the same thing.

| Mode | Parallel arenas | Simulator workers | Learning batch | Work/rest pacing |
| --- | ---: | ---: | ---: | --- |
| `background` | 96 | 4 | 256 | 35% work target per actor and learner, with lower CPU priority |
| `full` | 256 | All available CPU cores (14 on the reference Mac) | 512 | Uncapped |

Both modes use the Apple GPU for batched inference and learning. The background
mode adds deliberate, interruptible rest to both workers; it is a work/rest
budget, not a GPU-utilization percentage or temperature limit. Full mode runs
without that rest and keeps the Mac awake. Actual utilization depends on the
balance of simulation and learning; more arenas alone do not guarantee
higher throughput. The architecture and replay capacities are the same.

Each command starts one trainer in the background and resumes `runs/blacklight/latest.pt`.
An interactive terminal also shows compact inline progress. Ctrl-C closes that
display while training continues. `--no-watch` keeps the launch detached.
The older `trainctl` script remains available for existing launchers.

After each restart, the saved network resumes while its temporary replay buffer
refills. With the default settings, learning waits for 20,000 collected positions,
10,000 positions with ready training targets. Surviving positions normally need another 60 decisions in their
own arena (three simulated seconds) before their targets are ready. The views show
both requirements; overall warmup readiness follows the slower one.

The viewer stays open across `stop` and mode changes: it clears the old game when
training stops and follows fresh actor samples when the next run resumes.

To switch an already running trainer, save and stop it, then choose the new mode:

```bash
./blacklight stop
./blacklight train
```

`./blacklight status` reports the current run; `./blacklight log` follows its log.
Explicit trainer flags override a mode's defaults, for example
`./blacklight train --games 64 --duty-cycle 0.25`. `uv run arma-teach --help`
lists the controls. The viewer shows the actual resolved configuration and
optimizer hyperparameters, including overrides.

The actor plays concurrent simulator rounds and samples legal moves directly from
the policy. A clipped actor-critic loss reinforces executed actions whose returns
beat their recorded value estimates and discourages worse actions. Replay stores
each action's sampling probability, rejects searched decisions and limits policy
age with `--max-policy-lag` (default 64 updates). `--policy-clip` defaults to 0.2
and `--entropy-coef` to 0.01. Value and auxiliary heads still learn from replay.
It mixes self-play,
a live exploiter and the simulator's space fighter. V4 explores all
six actions. Temporary worker weights are published while RL runs; the saved
checkpoint is `latest.pt`, and `snapshots/` keeps a copy of it once per
`--snapshot-every` updates (default 1000) to fall back on. Closing the optional
viewer never stops training.

By default, half of the duels start in a close encounter: parallel riding,
crossing paths or offset head-on approaches. Distances, timing, position,
reflection and heading vary. These use normal physics and resources; the
remaining duels and all four-player games keep the normal arena spawns. The
purpose is to give the policy more fighting practice alongside full-round
strategy. Set `--combat-prob 0` for standard spawns only, or choose another
fraction up to 1. Three rounds in ten are against the space fighter
(`--hunter-prob`), and the outcome is discounted by 0.997 per decision
(`--gamma`), so a win ten seconds away is worth 55% of an immediate one and the
fighter is paid for finishing, not for surviving.

The space fighter retains its own built-in opponent logic; it never supplies
Blacklight's action labels. The old search flags (`--sims`, `--macro`,
`--search-prob`, `--gumbel`) are removed from the trainer. Resuming an older
search-trained checkpoint preserves its weights and counters and resets optimizer
moments for the new objective. Subsequent direct-RL sessions resume those moments.

The simulator's standalone tree inspection remains available for algorithm studies:
`uv run python -m arma_ai.search_trace --output runtime/search-trace.json` records
one isolated search using the current checkpoint. It writes no replay or weights
and is separate from training.

## One Blacklight checkpoint, every training session

The current V4 + 256-unit memory architecture is selected and frozen. Improve
opponents, scenarios and training while keeping this brain and checkpoint. See
[the architecture decision](docs/BLACKLIGHT-ARCHITECTURE-FREEZE.md) for the remaining
qualification work and the evidence required before any structural change.

Blacklight uses the V4 architecture. `runs/blacklight/latest.pt` is its one resumed training state; `snapshots/` keeps dated copies of it to fall back on. `background`, `full`, and
`imitate` all resume its current weights and cumulative update count, then save
back to that same file. There is no automatic fresh model, timestamped imitation
candidate, best-model rollback, previous-checkpoint rotation, or historical pool.
Parallel games generate experience for the learner; an exclusive lock prevents
two learners from overwriting V4. RL worker weight files exist only while that
trainer runs and are removed when it stops.

```bash
imitate                  # continue learning from strong bots until stopped
stop                     # save all completed learning and stop either method
imitate --epochs 3       # optionally do three additional cycles, then stop
```

`imitate` runs in the foreground and Ctrl-C also saves and stops it. Each cycle
collects 32 varied teacher lives and performs 120 learning updates. The example
cache is replaced each cycle; it does not create more model files. Checkpoints
are saved after learning cycles, periodically during long cycles, and on stop.
Run `watch` or `status` to see the same V4's progress. Evaluation is separate and
does not interrupt learning to run an automatic tournament.

The training UI uses one updating line: current stage, V4 update, a progress bar,
work counts and elapsed time. It cycles through recording, reading, learning,
checking and saving without printing a new line at each save. Wider terminals
also show the latest saved update. `background`, `full` and `watch` use the same
style; RL warmup follows the slowest readiness requirement, and its learning bar
counts updates toward the next report. The optional `dashboard` retains its
detailed charts.

Imitation sessions restore their optimizer state as well as weights. Switching
between RL and imitation preserves the weights and counters and resets optimizer
moments belonging to the other objective. Stop the current method before starting
another. If a previously trained run loses its main checkpoint, training reports
an error instead of silently creating a random fighter.

Teachers use the strongest available built-in profile at the maximum supported
AI_IQ setting of 100. A single recorded teacher fights one or three ordinary bots,
with varied arena sizes. Inputs precede the teacher's executed legal controls at
0.05-second decision boundaries. Whole teacher lives are separated for validation;
exact duplicate trajectories, missing provenance and truncated recordings are
rejected. Reports include turn recall and false turns. The stock bots never brake,
so they supply no braking demonstrations. Imitation learns contiguous sequences
with burn-in and checks whole-life memory. It supervises policy actions while
anchoring value, auxiliary and ownership predictions to their starting outputs;
actual outcome targets train those heads during RL.

`imitate --help` lists settings. `--data` accepts existing demonstration files;
`--run` explicitly selects a different run for development or testing. Bare
commands default to the same V4. The package entry point is `uv run arma-imitate`.

### Varied native bot candidates

`imitate --bot-pool diverse` records explicitly selected native profiles, including
four close-combat-biased candidates, the original wall and Quake presets, a path
hunter and a defensive tracer. Each 16-life block gives 12 lives to the fighter
teachers and four to the other styles, split equally between duels and four-player
games. Opponent styles vary too. The catalog is in `arma_ai/bot_profiles.py`.
These are candidates, not an established ranking of the best configurations.
The default remains `--bot-pool stock` until fighting qualification selects a recipe.

All six custom training profiles use search range 20, with reaction, emergency
handling, close-combat, pathfinding, enemy handling and trace detection at 10.
Wall-following settings, opening timings and starting modes retain each profile's
distinct settings.

The engine loads a private catalog and applies exact per-role names before cycles
spawn. It rejects unknown or incomplete rosters instead of silently falling back
to IQ selection. Recordings store the actual profile values and catalog hash.
The three `holdout_*` configurations never enter the diverse training schedule.

Evaluate frozen Blacklight against one or three named opponents with
`uv run arma-eval --builtin-profile holdout_fighter --rounds 20`; repeat the
`--builtin-profile` option to add opponents. Recording also accepts
`--teacher-profile fighter_cutter --opponent-profile stock_wall`.

This expands native demonstrations and evaluation. Simulator RL still uses
self-play, Hunter and the live exploiter. Stock-controller demonstrations contain
no braking; braking and corrective recovery practice still require the RL and
learner-state portions of the training recipe. Profile settings alone do not
establish a close-combat win rate or immediate aggression.

## Watching it play

```bash
uv run arma-show
```

Runs a separate policy demonstration in a headless real engine (not a training arena) and draws it in the terminal: every cycle's trail in its own colour, with a scoreboard of wins. `--bots 1 --ais 1` pits it against the game's best bot, `--ais 2 --bots 2` makes a four-player game, `--size 0` uses the full-size arena, `--speed 4` plays at four times real speed and `--walls` sets the trail length in metres (600 by default everywhere here, including training; the game's own default, -1, is endless). Ctrl-C quits.

## Watching live training

```bash
uv run arma-brain --no-open
```

Open the local URL printed by that command. **Live game** follows one selected
bot in an actual arena of the running trainer. Use the arena/player selectors to
choose which concurrent game and bot to observe. **Follow bot** and **Whole arena**
change only the camera. The browser does not run a separate match, advance the
simulator, play a recording or change training speed. A slow browser receives the
latest sample instead of a queue of old frames. When training stops or its feed
becomes stale, the game view clears and reports that state.

The selected bot's last executed move, speed, brakes, kills and weight update
come from the actor. **What the bot sees** displays one of its actual input maps,
captured with the scalar inputs at that executed root decision. Search branches
are never substituted for the played game. Opponents use their own trail colors;
those are other players' real trails, not alternative predicted routes.

**Network** shows the architecture reported by the loaded checkpoint and explains
each selected stage. Input stages show the selected bot's real observations; policy
and value stages show the actor's recorded outputs. Intermediate activation tensors
are not streamed. A checkpoint published on disk can be newer than the weights
used for the last move; the bot panel names the actual update used.

**Training** shows policy/value and auxiliary losses, learning rate, entropy,
gradient size, throughput, replay usage and training matchup counts. Each metric
includes a plain-language explanation. Charts identify their report update/time;
old losses are labelled as saved reports when training is stopped. The complete
hyperparameter/resource table uses the trainer's resolved values, falling back to
saved checkpoint configuration before an updated trainer has run.

Viewing is optional. The live feed contains periodic actual actor samples, so
intermediate decisions may occur between displayed samples. There are no replay,
freeze, step or speed controls in this training view.

## Playing against it in the game itself

The trained network also lives inside the game, so no Python is needed to play it: open Armagetron Advanced, choose Play Game, Local Game, and the first AI opponent is the network (named Brain; with several AIs in the game, the AI team is named after it). Game Setup has a "Neural opponents" setting for how many of the AI players it drives; the rest use the game's built-in AI. The weights are the file `brain/policy.bin` in this repository, exported from a checkpoint with

```bash
uv run arma-export                      # runs/blacklight/latest.pt -> brain/policy.bin
```

Rebuild the app after exporting, or drop the file into `~/Library/Application Support/Armagetron Advanced/brain/policy.bin`, which the game prefers over the bundled one. `NEURAL_SAMPLE 1` in the console makes it draw moves from its probabilities instead of always taking the best one, `NEURAL_NAME` renames it. The C++ side is `src/tron/gNeuralNet.cpp`; `tests/test_cpp_brain.py` checks it against PyTorch.

## Playing against it from Python

```bash
uv run arma-play
```

This starts a local server whose opponent is driven by the latest checkpoint. Open Armagetron Advanced, go to Play Game, Multiplayer, Custom Connect, and connect to `127.0.0.1` port 4534. Options: `--neural 3` for three network opponents, `--builtin 1` to add built-in bots, `--size 0` for the standard 500 m arena (the default `-3` is the single-player size).

The default single-opponent controller prewarms its inference path before starting
the server, preserving weights, sampling RNG and per-round history. Its configured
decision interval is 50 ms (20 decisions per second), independent of rendering FPS.
On the development Mac, a real-engine check of update 11,280 sustained that schedule.
With a separate 3024×1964, 4×-MSAA game renderer active, all 120 measured server responses
were under 50 ms (p95 11.45 ms, maximum 14.72 ms), with no skipped decision slots.
The [timing report](runs/blacklight/audit/controller-latency.json) separates setup,
warmed inference, real-engine pacing, and concurrent graphics load. It does not
reconstruct the timing of earlier human matches.

## Measuring it

```bash
uv run arma-eval --rounds 400 --size -3
```

Greedy duels against the strongest built-in AI or another checkpoint; results are appended to `runs/blacklight/eval.jsonl`.

Each result also includes `combat`: credited kills, deaths, rounds with a credited
kill and a surviving policy, wins with and without kill credit, and average win
duration. These counts cover completed rounds only, including when a time budget
ends a check early. The engine awards kill credit through enemy influence; this
records enemy deaths attributed to the policy, but does not establish intent or
prove that every uncredited win was passive. Judge fighting quality
alongside wins, draws and losses against fixed opponents on several arena sizes.

The network, learning targets and verification are described in [docs/V4.md](docs/V4.md).

## Building the engine

The engine has to live in a path without spaces (autotools breaks on them). From the repository root, the first time:

```bash
./bootstrap.sh && mkdir -p build-dedicated && cd build-dedicated && ../configure --enable-dedicated --with-boost=/opt/homebrew --disable-sysinstall --disable-useradd --disable-initscripts --disable-etc PKG_CONFIG_PATH=/opt/homebrew/opt/libxml2/lib/pkgconfig:/opt/homebrew/lib/pkgconfig LDFLAGS=-L/opt/homebrew/lib CPPFLAGS=-I/opt/homebrew/include
```

After that, and after every engine change:

```bash
cd build-dedicated && make -j12 && make install DESTDIR=$PWD/stage
```

Don't reinstall while training is running; a replaced binary can take the running engines down with it.
