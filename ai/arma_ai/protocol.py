"""Wire protocol between the patched Armagetron engine and the Python brain.

Transport: one Unix-domain stream socket per engine process. Python listens, every engine
connects at startup. All integers are little-endian.

Every message starts with a 12-byte header: magic (u32 'ARMA'), type (u32), payload length (u32).

Engine -> Python
  HELLO   u32 engine_id, u32 protocol, u32 n_slots, u32 grid, u32 n_local_planes,
          u32 n_global_planes, u32 n_scalars, f32 local_cell, f32 decision_interval
  STEP    u32 round_id, u32 tick, f32 round_time, u8 round_over, u8 n_alive (all cycles),
          u8 n_total (cycles that started the round), u8 n_slots,
          then per slot: u8 flags, u8 kills, u8 action_mask, u8 pad,
          and if FLAG_NEEDS_ACTION: local[GRID*GRID] u8, global[GRID*GRID] u8, scalars[N] f32

Python -> Engine
  ACTIONS u8 n_slots, then n_slots x u8 action (ignored for slots that did not ask)

The engine blocks after every STEP until the matching ACTIONS arrives (lockstep).
"""

from __future__ import annotations

import struct

PROTOCOL_VERSION = 1
MAGIC = 0x414D5241  # b"ARMA" little-endian

MSG_HELLO = 1
MSG_STEP = 2
MSG_ACTIONS = 3
MSG_WORLD = 4  # only with NEURAL_SPECTATE: every cycle's position, before each tick's STEP

GRID = 64
N_LOCAL_PLANES = 8
N_GLOBAL_PLANES = 8
N_SCALARS = 96
N_ACTIONS = 4

ACTION_STRAIGHT = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_BRAKE = 3
ACTION_NAMES = ("straight", "left", "right", "brake")

FLAG_ALIVE = 1 << 0
FLAG_NEEDS_ACTION = 1 << 1
FLAG_DIED = 1 << 2  # died since the previous STEP
FLAG_WON = 1 << 3  # set on the round_over STEP for the surviving winner(s)
FLAG_SPAWNED = 1 << 4  # first STEP of a new life

HEADER = struct.Struct("<III")
HELLO = struct.Struct("<IIIIIIIff")
STEP_HEAD = struct.Struct("<IIfBBBB")
SLOT_HEAD = struct.Struct("<BBBB")

OBS_BYTES = GRID * GRID * 2 + N_SCALARS * 4

# WORLD: u32 round_id, f32 round_time, u8 round_over, u8 step_follows, f32 x 4 arena bounds (low x, low y,
# high x, high y), u8 n, then per cycle: u16 player id, u8 slot (255 = built-in AI), u8 alive,
# f32 x, y, dir x, dir y, speed, 16-byte name. Without a following STEP the viewer answers with an
# empty ACTIONS message.
WORLD_HEAD = struct.Struct("<IfBB4fB")
WORLD_CYCLE = struct.Struct("<HBB5f16s")
