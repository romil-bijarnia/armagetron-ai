# Armagetron AI

A deep reinforcement learning agent for Armagetron Advanced that learns by playing against itself inside the real game engine.

The game itself does the simulation. The patched Armagetron engine in the rest of this repository adds a neural network bridge: chosen AI cycles stop using the built-in bot logic and instead send what they see to this Python project over a Unix socket every 0.05 s of game time, then execute the turn or brake the network picks. For training, the dedicated server runs headless in lockstep, so game time only advances when the network has answered and dozens of arenas run far faster than real time. The physics (rubber, wall acceleration, turn delay, brakes) are exactly the game's, because they are the game.

## What the network sees

Every decision gets three inputs, all rotated so the cycle always faces up:

- a local map, 64×64 cells of 2 m around the cycle (40 cells ahead, 24 behind), with separate planes for walls, its own trail, enemy trails, team trails, enemy heads, where each nearby enemy will be within the next second, and the outside of the arena;
- a global map of the whole arena at 64×64;
- 96 exact numbers: speed, rubber, brake reservoir, turn timing, 16 ray distances with hit types, and position, heading and speed of the three nearest enemies.

It answers with one of four actions: straight, left, right, brake. Turns are masked while the game's turn delay is running, so the network only ever issues moves a human could.

## Setting up

Install the Python environment once, from this folder. If the repository sits in an iCloud-synced folder (Desktop or Documents), keep the environment outside it: iCloud marks files it manages as hidden, and Python 3.13 ignores hidden `.pth` files, which breaks the `arma-*` commands.

```bash
uv venv --python 3.13 ~/.venvs/armagetron-ai && ln -s ~/.venvs/armagetron-ai .venv && uv sync
```

## Training

```bash
./trainctl start
```

`./trainctl start` trains in the background with the Mac kept awake and restarts itself from the last checkpoint if it ever crashes. `./trainctl gentle` does the same with 6 engines at low priority, for when you're using the laptop. `./trainctl status` shows whether it's running and how far it has got, `./trainctl watch` opens live charts of its progress in the terminal, `./trainctl log` follows the log, and `./trainctl stop` stops it after saving a checkpoint. Any extra flags go to the trainer, for example `./trainctl start --lr 1e-4`; `uv run arma-train --help` lists them all. To run the trainer in the foreground instead, use `uv run arma-train` (Ctrl-C stops it and saves).

An actor process runs 24 engines and picks moves; a learner process runs PPO on the GPU; rollouts move between them through shared memory. Arenas are mixed: self-play duels in three arena sizes, four-player free-for-alls, and matches against the game's strongest built-in AI. Some self-play opponents are frozen snapshots of earlier versions (saved to `runs/main/pool/`) so the policy can't forget how to beat old strategies. Training resumes from `runs/main/latest.pt` automatically. Progress goes to `runs/main/train.log` and `runs/main/metrics.jsonl`; the `win_vs_ai_*` columns are the share of rounds won against the built-in AI.

Reward is +1 for winning a round, up to −1 for dying (scaled by how many opponents were still alive) and +0.1 per kill.

## Watching it play

```bash
./trainctl show
```

Runs a real match in a headless engine and draws it live in the terminal: every cycle's trail in its own colour, with a scoreboard of wins. `--bots 1 --ais 1` pits it against the game's best bot, `--ais 2 --bots 2` makes a four-player game, `--size 0` uses the full-size arena and `--speed 4` plays at four times real speed. Ctrl-C quits.

## Watching it think

```bash
./trainctl brain
```

Opens a page in the browser that draws the network itself while it plays a live headless match: every layer as a slab of neurons in 3D (the two map towers, the number branch, the trunk and the move it picks), each neuron lit by how strongly it fires at that moment, and the strongest connections between layers glowing where signal flows. The 64×64 input maps show exactly what the AI sees, a small view of the arena shows the match, and the four move probabilities and the value estimate update with every decision.

Space pauses the match and the right arrow steps it one decision at a time, so a single choice can be studied; `[` and `]` change the speed. Hovering any neuron says what it is (for the inputs, which number or map cell it is and its value). Drag to turn the picture, scroll to zoom; Flat turns it side-on into the classic layer diagram, Glow switches the bloom off. Run it while training and it follows the newest weights after every update, flashing the connections that changed. Takes the same match flags as `show` (`--ais`, `--bots`, `--size`, `--speed`, `--checkpoint`).

## Playing against it

```bash
uv run arma-play
```

This starts a local server whose opponent is driven by the latest checkpoint. Open Armagetron Experimental, go to Play Game, Multiplayer, Custom Connect, and connect to `127.0.0.1` port 4534. Options: `--neural 3` for three network opponents, `--builtin 1` to add built-in bots, `--size 0` for the standard 500 m arena (the default `-3` is the single-player size), `--checkpoint runs/main/pool/u000200.pt` for an older version.

## Measuring it

```bash
uv run arma-eval --rounds 400 --size -3
uv run arma-eval --opponent runs/main/pool/u000100.pt
```

Greedy duels against the strongest built-in AI or another checkpoint; results are appended to `runs/main/eval.jsonl`.

The full system design, including planned changes and the test plan, is in [docs/DESIGN.md](docs/DESIGN.md).

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
