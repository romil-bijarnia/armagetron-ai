# Armagetron AI

A deep reinforcement learning agent for Armagetron Advanced that learns by playing against itself inside the real game engine.

The game itself does the simulation. The patched Armagetron engine in the rest of this repository adds a neural network bridge: chosen AI cycles stop using the built-in bot logic and instead send what they see to this Python project over a Unix socket every 0.05 s of game time, then execute the turn or brake the network picks. For training, the dedicated server runs headless in lockstep, so game time only advances when the network has answered and dozens of arenas run far faster than real time. The physics (rubber, wall acceleration, turn delay, brakes) are exactly the game's, because they are the game.

## What the network sees

Every decision gets three inputs, all rotated so the cycle always faces up:

- a local map, 64×64 cells of 2 m around the cycle (40 cells ahead, 24 behind), with separate planes for walls, its own trail, enemy trails, team trails, enemy heads, where each nearby enemy will be within the next second, and the outside of the arena;
- a global map of the whole arena at 64×64;
- 96 exact numbers: speed, rubber, brake reservoir, turn timing, 16 ray distances with hit types, and position, heading and speed of the three nearest enemies.

It answers with one of four actions: straight, left, right, brake. Turns are masked while the game's turn delay is running, so the network only ever issues moves a human could.

## Training

```bash
uv run arma-train --run runs/main
```

An actor process runs 24 engines and picks moves; a learner process runs PPO on the GPU; rollouts move between them through shared memory. Arenas are mixed: self-play duels in three arena sizes, four-player free-for-alls, and matches against the game's strongest built-in AI. Some self-play opponents are frozen snapshots of earlier versions (saved to `runs/main/pool/`) so the policy can't forget how to beat old strategies. Training resumes from `runs/main/latest.pt` automatically. Progress goes to `runs/main/train.log` and `runs/main/metrics.jsonl`; the `win_vs_ai_*` columns are the share of rounds won against the built-in AI.

Reward is +1 for winning a round, up to −1 for dying (scaled by how many opponents were still alive) and +0.1 per kill.

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
