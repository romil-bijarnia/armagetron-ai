#!/usr/bin/env bash
# Native SDL/OpenGL app. Preview mode renders an isolated match without a window.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-run}"
BUILD_DIR="$PROJECT_DIR/build-macos"
APP_BUNDLE="/Applications/Armagetron Advanced.app"
cd "$PROJECT_DIR"

case "$MODE" in
  --preview)
    shift
    if [[ ! -f "$BUILD_DIR/Makefile" ]]; then
      mkdir -p "$BUILD_DIR"
      (cd "$BUILD_DIR" && "$PROJECT_DIR/desktop/os-x/configure_for_bundle.sh")
    fi
    make -C "$BUILD_DIR" -j8
    if [[ ! -d "$APP_BUNDLE/Contents/Resources/armagetronad" ]]; then
      desktop/os-x/build_native_app.sh "$BUILD_DIR"
    fi
    # The capture helper takes --output and optional camera/video overrides.
    exec "$PROJECT_DIR/ai/.venv/bin/python" desktop/os-x/capture_preview.py \
      --client "$BUILD_DIR/src/armagetronad_main" "$@"
    ;;
  run|--verify|--logs|--telemetry|--debug) ;;
  *) printf 'usage: %s [run|--verify|--logs|--telemetry|--debug|--preview --output PATH ...]\n' "$0" >&2; exit 2 ;;
esac

pkill -x armagetronad 2>/dev/null || true
desktop/os-x/build_native_app.sh "$BUILD_DIR"
if [[ "$MODE" == --debug ]]; then
  exec lldb -- "$APP_BUNDLE/Contents/MacOS/armagetronad"
fi
/usr/bin/open -n "$APP_BUNDLE"
case "$MODE" in
  --verify) sleep 2; pgrep -x armagetronad >/dev/null ;;
  --logs|--telemetry)
    exec /usr/bin/log stream --info --style compact --predicate 'process == "armagetronad"'
    ;;
esac
