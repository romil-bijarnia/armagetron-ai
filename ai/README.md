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

Runs a real match in a headless engine and draws it live in the terminal: every cycle's trail in its own colour, with a scoreboard of wins. `--bots 1 --ais 1` pits it against the game's best bot, `--ais 2 --bots 2` makes a four-player game, `--size 0` uses the full-size arena, `--speed 4` plays at four times real speed and `--walls` sets the trail length in metres (600 by default everywhere here, including training; the game's own default, -1, is endless). Ctrl-C quits.

## Watching it think

```bash
./trainctl brain
```

Opens a page in the browser that shows what the network thinks while it plays a live headless match. The arena fills the left of the screen with the match drawn on it and the network's beliefs painted over the floor: who will hold each cell two seconds from now (its ownership head, solid where it is sure), or, with Floor set to "now", who gets to each cell first as the engine computes it. The sixteen sensor rays fan out from the cycle. Beside it: the odds of each move with the one taken, its win chance (the value head), how likely it thinks it is dead within two seconds and the territory share it expects, each with its recent history, plus what it feels (speed, rubber, brake, enemies). To the right, the four 64×64 maps exactly as the network sees them and the 96 exact numbers; hover a cell for what is in it.

Attention paints where a cell of the arena view looks in the v3 attention layers (hover the arena to pick the cell; the cell my head is in otherwise; keys 1 and 2 pick the layer) and shows every head of both layers small. Scroll down for Inside: every stage of both towers, the number branch and the trunk drawn as the pictures they are, one tile per filter lit by how strongly it fires; hover a tile to see that filter large. Space pauses the match and the right arrow steps it one decision at a time, so a single choice can be studied; `[` and `]` change the speed. Run it while training and it follows the newest weights after every update, a bar under each stage showing how much that update changed it, and the numbers page below shows the training itself: win rates, round length, entropy, KL, losses, gradient norm, learning rate and speed, each with its history. Takes the same match flags as `show` (`--ais`, `--bots`, `--size`, `--speed`, `--walls`, `--checkpoint`).

## Playing against it in the game itself

The trained network also lives inside the game, so no Python is needed to play it: open Armagetron Advanced, choose Play Game, Local Game, and the first AI opponent is the network (named Brain; with several AIs in the game, the AI team is named after it). Game Setup has a "Neural opponents" setting for how many of the AI players it drives; the rest use the game's built-in AI. The weights are the file `brain/policy.bin` in this repository, exported from a checkpoint with

```bash
uv run arma-export                      # runs/main/latest.pt -> brain/policy.bin
```

Rebuild the app after exporting, or drop the file into `~/Library/Application Support/Armagetron Advanced/brain/policy.bin`, which the game prefers over the bundled one. `NEURAL_SAMPLE 1` in the console makes it draw moves from its probabilities instead of always taking the best one, `NEURAL_NAME` renames it. The C++ side is `src/tron/gNeuralNet.cpp`; `tests/test_cpp_brain.py` checks it against PyTorch.

## Playing against it from Python

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
