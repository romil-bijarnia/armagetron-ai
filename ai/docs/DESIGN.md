# Armagetron AI: System Design

## 1. Goal and constraints

The goal is a neural network agent that plays Armagetron Advanced as well as possible, learns by playing against itself on the game's real physics, and can be played against in the normal game client. Everything runs on one machine: an M3 Max MacBook Pro with 14 CPU cores (10 performance), a 30-core GPU and 36 GB of unified memory. Training has to be resumable at any point, and it needs a low-impact mode so the laptop stays usable while it runs.

The central design decision is to use the game itself as the simulator rather than a re-implementation. Armagetron's movement model (rubber, wall acceleration, turn delay, turn speed loss, brakes, explosions punching holes into walls) is intricate, and an agent trained on an approximation would learn habits that fail in the real game. A small patch turns the game's dedicated server into a fast, headless training environment, so there is no gap between what the agent trains on and what a human plays against.

Current state: the full pipeline is built and has been run. The latest checkpoint is update 56 (about 1.84 million decisions of experience), with past-self snapshots at updates 20 and 40. At update 20 it already won 60 of 60 greedy small-arena duels against the game's strongest built-in AI.

## 2. Architecture overview

```
 ai/ (Python)                                             Patched Armagetron (repository root)
 +----------------------------------+                     +---------------------------------------+
 | Learner process                  |                     | armagetronad-dedicated  x N engines    |
 |   PPO updates on the GPU         |                     |   lockstep clock, runs without humans |
 |   publishes weights (actor.pt)   |                     |   gNeural bridge drives K AI slots    |
 |        ^  rollouts via 2 shared  |   unix socket       |   built-in AIs fill the other slots   |
 |        |  memory buffers         |   STEP (maps,       |                                       |
 | Actor process                    | <-- scalars, flags) |                                       |
 |   EnginePool + inference ------- | --> ACTIONS ------> |                                       |
 +----------------------------------+                     +---------------------------------------+

 Play mode: arma-play starts one real-time server plus a Brain; the human joins with the stock client.
```

Each engine process is one arena. It hosts K neural-controlled cycles and any number of the game's own AI cycles, advances game time only as fast as the network answers, and sends one STEP message per decision tick. The actor process multiplexes all engines over Unix sockets, batches every pending decision into one GPU call, and records the transitions. The learner process trains on finished rollouts and publishes new weights, which the actor picks up after each rollout.

## 3. Engine side

### 3.1 Neural bridge (`src/tron/gNeural.cpp`)

The bridge takes over the first `NEURAL_SLOTS` AI players in join order. For those players the built-in bot logic is skipped (`gAIPlayer::Timestep` and `RightBeforeDeath` return early), so the cycle only moves when the network says so.

Decisions happen every `NEURAL_DECISION_INTERVAL` seconds of game time (0.05 s, i.e. 20 decisions per second), called from inside the world timestep right after all objects have moved. Each tick the bridge takes a census of all cycles: who is alive, how many teams are still alive, who died since the last tick and who gets the kill. Kill credit uses the same enemy-influence rule as the game's own scoring (exposed as `gCycleMovement::Hunter()`), so a cycle that drives into the rim on its own is a suicide, not a kill for the nearest opponent. A round counts as over once at most one team has a living cycle.

The bridge then sends a STEP, blocks until the ACTIONS reply arrives and applies each action through `gCycle::Act`, which is exactly the path a human key press takes: turn left, turn right, brake on or off. Turns are masked out while the game would not execute them immediately (turn delay running or turns already queued). This keeps the agent to moves a human could make, and stops it from queuing turns that would fire at positions it didn't choose.

Settings: `NEURAL_SOCKET`, `NEURAL_SLOTS`, `NEURAL_ENGINE_ID`, `NEURAL_DECISION_INTERVAL`, `NEURAL_LOCAL_CELL`, `NEURAL_END_ROUND_WITHOUT_NEURAL` (training: end the round as soon as no neural cycle is left, instead of simulating the built-in bots to the end), `NEURAL_CONTROL_AFTER_ROUND` (play mode: keep steering after the winner is decided) and `NEURAL_DEBUG`.

### 3.2 Headless lockstep mode

A stock dedicated server refuses to play without a connected client: it naps, stops matches and ends rounds at once. `PLAY_WITHOUT_HUMANS` bypasses the four checks responsible.

`LOCKSTEP_DT` replaces the wall clock with a simulated one. Every frame advances game time by exactly that step, and `tDelay` and the network `select` never sleep. The step is 0.025 s, which matches the dedicated server's own physics granularity (0.9 / `DEDICATED_FPS`). One engine plays about 17,000 to 20,000 decisions per second this way, hundreds of times faster than real time.

Arena setup has two traps the configuration handles. A server with no humans uses the single-player `SP_` settings, so every setting is written in both forms. And the game normally puts every AI into one shared AI team, which makes AI-only rounds end before they start. Instead, `TEAMS_MIN` is set to the number of AIs with `TEAM_MAX_PLAYERS 1` and team balancing on, so every cycle gets its own team. The balancer always picks the strongest built-in AI character. `FINISH_TYPE 3` stops AI-only rounds from being cut short.

### 3.3 Wire protocol

Little-endian, every message prefixed by magic `ARMA`, type and payload length.

| Message | Direction | Content |
|---|---|---|
| HELLO | engine to Python | engine id, protocol version, slot count, grid size, plane counts, scalar count, local cell size, decision interval |
| STEP | engine to Python | round id, tick, round time, round-over flag, cycles alive, cycles at round start, then per slot: flags (alive, needs action, died, won), kills since last step, action mask, and if an action is needed the local map, global map and scalars |
| ACTIONS | Python to engine | one action byte per slot |

## 4. Observation

Every observation is egocentric: both maps are rotated so the cycle always drives up the image. This means the network never has to learn the same situation four times over for the four headings.

The local map is 64 by 64 cells of 2 m around the cycle, 40 cells ahead and 24 behind (80 m ahead, 64 m to each side). The global map covers the whole arena at 64 by 64, rotated about the arena centre. Each cell is one byte whose bits are separate planes, so one observation is 8 KB of maps plus 384 bytes of scalars.

| Bit | Local map plane | Global map plane |
|---|---|---|
| 0 | any wall | any wall |
| 1 | own trail | own trail |
| 2 | enemy trail | enemy trail |
| 3 | teammate trail | own position |
| 4 | enemy cycle positions | enemy cycle positions |
| 5 | where each nearby enemy will be within 1 s if it drives straight | outside the arena |
| 6 | outside the arena | teammate trail |
| 7 | teammate positions | teammate positions |

Walls are read from the engine's live wall lists and sampled at half-cell spacing, with each sample checked against the wall's real danger state. Holes blown by explosions, walls of dead cycles that have expired, and the part of a wall that dedicated servers extend ahead of a cycle for prediction all correctly show as empty.

The 96 scalars carry what a grid can't show precisely:

| Index | Content |
|---|---|
| 0 to 7 | speed relative to base speed, rubber used, brake reservoir, braking flag, time since last turn relative to the turn delay, can turn left, can turn right, queued turns |
| 8 to 15 | arena width and height, own position in the rotated frame, round time, enemies alive, teammates alive, speed factor |
| 16 to 47 | 16 rays at 22.5 degree steps (exact distance to the first wall), each encoded both linearly and as exp(-d/8) so close range is sharp |
| 48 to 63 | what the forward, right, back and left rays hit: rim, enemy, teammate or own trail |
| 64 to 87 | the three nearest enemies: relative position, distance, relative heading, speed, braking flag, presence flag |
| 88 to 95 | reserved |

The agent only sees what a human player could see on screen: walls, cycle positions, headings and speeds, and its own rubber and brake gauges. It doesn't see enemy rubber.

## 5. Actions

There are four discrete actions per decision: straight, left, right and brake (straight with the brake held). Left and right are masked while a turn would not execute immediately. At 20 decisions per second and the default 0.1 s turn delay, the agent can make a turn every second decision, which is enough for double turns and U-turns.

## 6. Reward

Winning a round gives +1. Dying costs between 0 and -1, scaled by the share of opponents still alive at that moment: dying first in a four-player free-for-all costs -1, and finishing second costs -1/3. Dying at the same moment as the last opponent is a draw and costs 0. Each kill the game credits adds +0.1. There is no survival bonus, because in Tron surviving is already what wins.

## 7. Learning system

### 7.1 Network

The network is a convolutional actor-critic with 1,908,869 parameters, roughly the size of DeepMind's original Atari DQN network. One decision costs about 35 million multiply-adds, which is about 1 ms for a batch of 24 cycles on the GPU.

| Part | Layers | Parameters |
|---|---|---|
| Local map tower | 8 planes in; 4x4 stride-4 patch conv to 64 channels (16x16); two 3x3 convs at 64; 3x3 stride-2 conv to 128 (8x8); 3x3 stride-2 conv at 128 (4x4); linear 2048 to 256 | 828,096 |
| Global map tower | same shape at half width: patch conv to 32, one 3x3 conv, stride-2 convs to 64; linear 1024 to 256 | 331,200 |
| Scalar MLP | 96 to 256 to 256 | 90,624 |
| Trunk | concatenated 768 to 512 to 512 | 656,384 |
| Policy head | 512 to 4 logits, invalid actions masked | 2,052 |
| Value head | 512 to 1 | 513 |

All activations are ReLU, with orthogonal initialisation and a small-gain policy head so early play is close to uniform. There's no recurrence: Tron is almost fully observable from the maps, so the current frame is enough.

The patch stem is a performance decision. Convolutions at full 64 by 64 resolution are very slow on Apple GPUs, and cutting each map into 4 by 4 patches with a strided convolution makes training about 15 times faster. Unlike pooling, the strided convolution still sees every bit of every patch, so nothing is thrown away before the first learned layer. Bit-planes are unpacked on the GPU with a lookup table into a contiguous tensor, because convolution backward on a permuted view is three times slower on MPS.

### 7.2 PPO

| Setting | Value |
|---|---|
| Rollout per update | 32,768 learner decisions |
| Epochs, minibatch | 2 epochs, 4,096 |
| Learning rate | 2.5e-4, Adam (eps 1e-5) |
| Discount, GAE lambda | 0.995 (about a 10 s horizon at 20 Hz), 0.95 |
| Clip, value clip | 0.2, 0.2 |
| Entropy, value coefficients | 0.01, 0.5 |
| Gradient norm clip | 0.5 |
| Early stop | mean KL above 0.03 in an epoch |

Every neural cycle is its own stream of experience. Transitions are linked per stream so GAE can run over a flat buffer. A transition whose outcome isn't known yet at the end of a rollout is carried into the next one instead of being cut off, so no episode is truncated by the rollout boundary.

### 7.3 Actor and learner processes

The actor runs the engines and makes every decision. The learner runs PPO. They exchange finished rollouts through two shared-memory buffers, so the actor fills one while the learner trains on the other. The actor reloads the learner's weights after each rollout, so its policy lags by at most two updates. PPO's importance ratio is computed against the policy that actually chose each move, so the lag is accounted for.

### 7.4 Opponents and arenas

Each engine runs a fixed arena type, and the mix makes the agent generalise across arena sizes and player counts. Within self-play arenas, every non-first slot has a 35% chance per round of being driven by a frozen past snapshot (taken every 20 updates) instead of the current learner. That keeps it from forgetting how to beat old strategies. Only the learner's decisions are trained on.

| Arena | Cycles | Size factor (arena side) | Engines of 24 |
|---|---|---|---|
| duel_small | 2 neural | -3 (177 m) | 6 |
| duel_mid | 2 neural | -1.5 (297 m) | 4 |
| duel_std | 2 neural | 0 (500 m) | 4 |
| ffa4 | 4 neural | -1 (354 m) | 4 |
| vs_ai_small | 1 neural, 1 built-in | -3 | 2 |
| vs_ai_std | 1 neural, 1 built-in | 0 | 2 |
| mixed_ffa | 2 neural, 2 built-in | -1 | 2 |

## 8. Evaluation

Strength is measured three ways. Greedy duels against the strongest built-in AI (`arma-eval`) were the first yardstick, and the agent saturated them almost immediately. League play (`arma-league`) pits the latest checkpoint against a spread of past snapshots and turns each win rate into an Elo gap. Rising Elo against fixed old snapshots is the signal that self-play is still making progress rather than cycling. Human games through play mode are the final test.

Training logs per update: decisions per second, collection and learning time, policy lag, mean episode length, policy and value losses, entropy, KL, clip fraction, explained variance, and the learner's recent score in each arena type.

## 9. Play mode

`arma-play` starts one dedicated server in real time with the bridge connected to a `Brain` (checkpoint plus greedy or sampled action choice). It waits for the human to join from the stock Armagetron Experimental client on 127.0.0.1:4534. The installed client and the patched server come from the same source revision, so they're compatible. Options choose the number of neural and built-in opponents, arena size, speed and checkpoint. An older snapshot or sampled moves make a weaker, less predictable opponent.

## 10. Measured performance and bottlenecks

| Measurement | Result |
|---|---|
| One engine, random moves | 17,000 to 20,000 decisions/s |
| Full pipeline, 24 engines, early (rounds of about 2 s) | about 3,000 decisions/s |
| Full pipeline, 24 engines, rounds of about 70 s | 800 to 1,600 decisions/s |
| Learner, one 4,096 minibatch forward and backward | 266 ms |
| Inference, batch of 24 cycles | about 1 ms |

Two bottlenecks show up. The first is engine-side observation cost. Every tick, every engine samples every wall in the arena and every neural cycle walks all samples, so the cost grows with total trail length. Collection slowed by roughly two thirds as the agent learned to survive longer, and this is the most likely cause. Learning also slowed in the same period, which points at contention as well, so both get measured before and after the fix. The second bottleneck is GPU sharing. The actor's small inference batches queue behind the learner's training kernels on the same GPU, so engines wait longer for answers while an update is running.

## 11. Planned changes

These are in priority order. None of them has been run yet.

**Incremental world raster.** Each engine keeps one uint16 grid at 1 m resolution over the arena (0.5 MB for the standard arena). Each cell holds a bitmask of whose trail occupies it: bits 0 to 14 are the round's cycle index and bit 15 is the rim. Finished wall segments never change, so each is rasterised once when it first appears. Only the handful of growing segments (one per living cycle, clipped at the cycle's position) are redrawn each tick. Rare events trigger a full rebuild from the wall lists: deaths (explosions punch holes and walls later expire) and finite wall length settings. The local map then becomes a rotated read of 128 by 128 raster cells pooled into 64 by 64, and the global map a pooled read of the whole raster. Own, teammate and enemy planes come from the bitmasks at read time. Per-agent cost becomes a fixed few thousand cell reads plus 16 rays, independent of how long the round has run, which should restore and exceed the early throughput.

**Resource modes.** An `overnight` profile runs 24 to 48 engines at normal priority with the Mac kept awake. A `background` profile runs about 6 engines with every process niced and no keep-awake lock, and caps the learner's GPU duty cycle by pausing between minibatches, so the window server and your apps stay smooth. GPU load is what usually makes a Mac's interface stutter, so the duty cap matters more than CPU priority.

**Less GPU contention.** First, try 48 engines so each inference call carries about twice as many decisions; with the raster fix the CPU has room for it. Then evaluate a leaner local tower (one 3x3 layer at width 48, about half the training cost) against the current one in the league, keeping it only if strength holds. A further option is running actor inference through Core ML on the Neural Engine, which is a separate accelerator and wouldn't compete with the learner at all.

**Prioritised past-self sampling.** Track the learner's score against each snapshot and sample opponents in proportion to (1 - win rate) squared, keeping about six snapshots loaded. Time then goes to the old strategies the agent still struggles with, rather than ones it already beats.

**Robustness curriculum.** Vary `SPEED_FACTOR` around the default, add team arenas (the observation already has teammate planes), and add other maps from the game's resource set, so human-chosen settings don't surprise it.

**Schedules.** Anneal the learning rate and entropy bonus once league Elo flattens.

**Optional safety lookahead in play mode.** Veto a chosen move if a short ray check shows a certain crash within the next decision while an alternative is safe. This is off by default and only for human games.

## 12. Test plan (for later)

| Area | Test |
|---|---|
| Python pipeline | existing: actor and learner round trip against fake engines; GAE linkage across streams |
| Frame and controls | automated version of the manual check: a left turn rotates heading anticlockwise, ray distances match the arena walls before and after, turns masked during the turn delay |
| Round logic | duel and four-player rounds end exactly when one team is left; kill credit matches the ladderlog `DEATH_FRAG` lines; suicides give no credit |
| Raster | after random play, the incremental raster equals a full rebuild cell for cell, including after deaths and wall expiry |
| Throughput | engines alone, full pipeline at 24 and 48 engines, early and long rounds, background profile's effect on the laptop |
| Strength | built-in duels at three arena sizes, league Elo every 50 updates, human games |
| Play mode | the stock client connects, the neural opponent moves in real time, keeps steering after the round is decided, and the server restarts cleanly between rounds |

## 13. Repository map

| Path | Purpose |
|---|---|
| `ai/arma_ai/protocol.py` | wire format constants and structs |
| `ai/arma_ai/engine.py` | `ArenaConfig` (engine settings) and `EnginePool` (process and socket management) |
| `ai/arma_ai/model.py` | `PolicyNet` |
| `ai/arma_ai/train.py` | `Actor`, `Learner`, PPO, arena mix |
| `ai/arma_ai/play.py` | `Brain` and the play server |
| `ai/arma_ai/evaluate.py`, `ai/arma_ai/league.py` | strength measurement |
| `ai/arma_ai/smoke.py` | engine throughput check and observation image dumps |
| `ai/tests/` | fake engine and pipeline tests |
| `ai/runs/main/` (not in git) | `latest.pt`, `actor.pt`, past-self `pool/`, `metrics.jsonl`, `train.log` |
| repository root (`src/`, `config/`, ...) | the patched Armagetron engine: `src/tron/gNeural.cpp/.h`, lockstep and no-human changes in `gGame.cpp`, `tSysTime.cpp`, `nSocket.cpp`, accessors in `gCycleMovement` |
