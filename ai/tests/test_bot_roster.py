"""Named roles and private character deployment cannot silently fall back to IQ matching."""

from dataclasses import asdict
from hashlib import sha256
from types import SimpleNamespace
import sys
import time

import pytest

from arma_ai import engine
from arma_ai.engine import AI_CHARACTERS_FILENAME, ArenaConfig, EnginePool


def characters(*names):
    return "# Private qualification roster\n" + "".join(
        f"{name} 10 10 16 30 10 10 10 10 10 10 3 0 1 Exact named role\n" for name in names)


def test_default_arena_keeps_stock_selection_and_serialization():
    arena = ArenaConfig()
    settings = arena.settings()
    assert not {"AI_CHARACTER_FILE", "NEURAL_CHARACTER_PROFILE", "BUILTIN_CHARACTER_PROFILES"} & settings.keys()
    assert ArenaConfig(**asdict(arena)) == arena


@pytest.mark.parametrize("opponents", [1, 3])
def test_named_roles_deploy_exact_private_bytes_per_engine(tmp_path, opponents):
    names = tuple(f"Opponent{i}" for i in range(opponents))
    text = characters("Teacher", *names)
    a = ArenaConfig(slots=1, builtin_ais=opponents, ai_characters=text,
                    controlled_profile="Teacher", builtin_profiles=names)
    b = ArenaConfig(slots=1, builtin_ais=1, ai_characters=characters("Other", "Counter"),
                    controlled_profile="Other", builtin_profiles=("Counter",))
    pool = SimpleNamespace(arenas=[a, b], workdir=tmp_path, base_port=48000, sock_path="/tmp/test-roster.sock")
    first, second = (EnginePool._write_config(pool, i) for i in range(2))
    path = first / "userconfig" / AI_CHARACTERS_FILENAME
    assert path.read_bytes() == text.encode()
    assert sha256(path.read_bytes()).hexdigest() == sha256(text.encode()).hexdigest()
    assert (second / "userconfig" / AI_CHARACTERS_FILENAME).read_text() == b.ai_characters
    assert path.read_text() == text  # writing another arena did not replace this roster
    settings = (first / "userconfig" / "autoexec.cfg").read_text()
    assert f"AI_CHARACTER_FILE {AI_CHARACTERS_FILENAME}\n" in settings
    assert "NEURAL_CHARACTER_PROFILE Teacher\n" in settings
    assert f"BUILTIN_CHARACTER_PROFILES {' '.join(names)}\n" in settings
    assert ArenaConfig(**asdict(a)) == a
    assert ArenaConfig(**{**asdict(a), "builtin_profiles": list(names)}) == a


@pytest.mark.parametrize("kwargs,match", [
    ({"controlled_profile": "bad name"}, "single-token"),
    ({"builtin_profiles": "Opponent"}, "sequence"),
    ({"builtin_profiles": ("One", "Two")}, "every ordinary"),
    ({"controlled_profile": "Missing", "ai_characters": characters("One", "Two")}, "missing"),
    ({"ai_characters": characters("Only")}, "unique provisioning"),
    ({"ai_characters": characters("Same", "Same")}, "duplicated"),
    ({"ai_characters": "Broken 10 10\n"}, "13 integer"),
    ({"ai_characters": "  # Unrecognized native comment\n" + characters("One", "Two")}, "column zero"),
    ({"ai_characters": "Bad 10 10 16 30 10 10 10 10 10 10 4 0 1\n"}, "unsupported"),
    ({"ai_characters": "Bad 10 10 65 30 10 10 10 10 10 10 3 0 1\n"}, "unsupported"),
    ({"ai_characters": "Bad 11 10 16 30 10 10 10 10 10 10 3 0 1\n"}, "unsupported"),
    ({"ai_characters": "Bad 10 10 -1 30 10 10 10 10 10 10 3 0 1\n"}, "13 integer"),
    ({"ai_characters": characters("One", "Two") + "\0"}, "control characters"),
])
def test_invalid_names_counts_and_character_properties_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ArenaConfig(slots=1, builtin_ais=1, **kwargs)


def test_explicit_roster_cannot_be_overridden_by_an_unrelated_extra_setting():
    arena = ArenaConfig(slots=1, builtin_ais=1, ai_characters=characters("Teacher", "Opponent"),
                        controlled_profile="Teacher", builtin_profiles=("Opponent",),
                        extra={"AI_CHARACTER_FILE": "aiplayers.cfg"})
    with pytest.raises(ValueError, match="conflict"):
        arena.settings()


def test_old_native_binary_is_rejected_before_process_or_socket_creation(tmp_path, monkeypatch):
    old = tmp_path / "old-engine"
    old.write_bytes(b"old engine with AI_CHARACTER_FILE but no named role support")
    monkeypatch.setattr(engine, "ENGINE_BIN", old)
    monkeypatch.setattr(engine.socket, "socket", lambda *a, **kw: pytest.fail("created socket before capability check"))
    monkeypatch.setattr(engine.subprocess, "Popen", lambda *a, **kw: pytest.fail("spawned unsupported engine"))
    arena = ArenaConfig(slots=1, builtin_ais=1, controlled_profile="Teacher", builtin_profiles=("Opponent",))
    with pytest.raises(RuntimeError, match="rebuild and stage"):
        EnginePool([arena], workdir=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_native_capability_is_read_once_per_pool_and_custom_commands_can_bypass(tmp_path, monkeypatch):
    executable = tmp_path / "engine"
    executable.write_bytes(b"NEURAL_CHARACTER_PROFILE\0BUILTIN_CHARACTER_PROFILES\0")
    monkeypatch.setattr(engine, "ENGINE_BIN", executable)
    monkeypatch.setattr(EnginePool, "_spawn", lambda *a: None)
    monkeypatch.setattr(EnginePool, "_accept_all", lambda *a: None)
    arena = ArenaConfig(slots=1, builtin_ais=1, controlled_profile="Teacher", builtin_profiles=("Opponent",))
    pool = EnginePool([arena, arena], workdir=tmp_path / "supported")
    try:
        assert pool.n == 2
    finally:
        pool.close()
    executable.unlink()
    pool = EnginePool([arena], workdir=tmp_path / "custom", command=lambda *a: ["test-double"])
    pool.close()


def tracked_pool():
    failed = {}

    class TrackedPool(EnginePool):
        def close(self):
            failed["pool"] = self
            super().close()

    return TrackedPool, failed


def assert_startup_released(pool):
    assert all(process.poll() is not None for process in pool.procs if process is not None)
    assert pool.listener.fileno() == -1
    assert pool.sel.get_map() is None
    assert not pool._sockdir.exists()
    assert not pool.workdir.exists()
    pool.close()  # evaluation pools may call close again after constructor failure


@pytest.mark.parametrize("missing_profile", [False, True])
def test_child_exit_before_hello_fails_quickly_and_reaps_every_process(missing_profile):
    Pool, failed = tracked_pool()
    message = ("neural bridge: NEURAL_CHARACTER_PROFILE is missing or duplicated in AI_CHARACTER_FILE"
               if missing_profile else "simulated native startup failure")

    def command(index, *_):
        code = "import time; time.sleep(30)" if index == 0 else f"import sys; print({message!r}, file=sys.stderr); sys.exit(73)"
        return [sys.executable, "-c", code]

    arena = ArenaConfig(slots=1, builtin_ais=1, controlled_profile="Missing", builtin_profiles=("Opponent",))
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="engine 1 exited 73 before HELLO") as error:
        Pool([arena, arena], command=command, log_engines=True)
    assert time.monotonic() - started < 2
    assert message in str(error.value)
    assert_startup_released(failed["pool"])


def test_partial_spawn_failure_cleans_already_started_sibling():
    Pool, failed = tracked_pool()

    def command(index, *_):
        return ([sys.executable, "-c", "import time; time.sleep(30)"] if index == 0
                else ["/definitely/missing/armagetron-test-engine"])

    with pytest.raises(FileNotFoundError):
        Pool([ArenaConfig(), ArenaConfig()], command=command)
    assert_startup_released(failed["pool"])


def test_incomplete_connected_hello_respects_global_timeout_and_closes_unregistered_socket():
    Base, failed = tracked_pool()

    class Pool(Base):
        def _accept_all(self, timeout=60):
            return super()._accept_all(timeout=.15)

    def command(_index, _root, sock_path, _arena):
        code = ("import socket,time; s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); "
                f"s.connect({sock_path!r}); s.sendall(b'AR'); time.sleep(30)")
        return [sys.executable, "-c", code]

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="did not complete HELLO"):
        Pool([ArenaConfig()], command=command)
    assert time.monotonic() - started < 2
    assert_startup_released(failed["pool"])


def test_partial_hello_survives_a_poll_timeout_without_losing_bytes():
    from arma_ai import protocol as P

    payload = P.HELLO.pack(0, P.PROTOCOL_VERSION, 2, P.GRID, P.N_LOCAL_PLANES,
                           P.N_GLOBAL_PLANES, P.N_SCALARS, P.N_MAPS, 2., .05)
    packet = P.HEADER.pack(P.MAGIC, P.MSG_HELLO, len(payload)) + payload

    def command(_index, _root, sock_path, _arena):
        code = ("import socket,time; s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); "
                f"s.connect({sock_path!r}); packet=bytes.fromhex({packet.hex()!r}); "
                "s.sendall(packet[:2]); time.sleep(.6); s.sendall(packet[2:]); time.sleep(30)")
        return [sys.executable, "-c", code]

    with EnginePool([ArenaConfig()], command=command) as pool:
        assert pool.conns[0] is not None
    assert_startup_released(pool)


def test_evaluation_deadline_override_also_cleans_failed_startup():
    from arma_ai.evaluate import _EvaluationPool

    failed = {}

    class Pool(_EvaluationPool):
        def close(self):
            failed["pool"] = self
            super().close()

    command = lambda *_: [sys.executable, "-c", "import time; time.sleep(30)"]
    started = time.monotonic()
    with pytest.raises((RuntimeError, TimeoutError)):
        Pool([ArenaConfig()], command=command, deadline=started + .15)
    assert time.monotonic() - started < 2
    assert_startup_released(failed["pool"])
