#!/bin/bash

# One command from a fresh checkout to a native macOS app for the chip of this Mac:
# configures against Homebrew, compiles, and packs the bundle with its libraries inside.
# Installs the single app in /Applications; build-macos keeps archives only.
#
# usage: desktop/os-x/build_native_app.sh [build directory, default build-macos]
# needs: brew install autoconf automake boost dylibbundler freetype ftgl glew libpng
#        libxml2 pkgconf protobuf sdl2 sdl2_image sdl2_mixer

set -euo pipefail

SRC=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-${SRC}/build-macos}

test -x "${SRC}/configure" || (cd "${SRC}" && ./bootstrap.sh)

mkdir -p "${OUT}"
OUT=$(cd "$OUT" && pwd)
cd "${OUT}"
test -r Makefile || "${SRC}/desktop/os-x/configure_for_bundle.sh"
make -j$(sysctl -n hw.ncpu)
./config.status desktop/os-x/build_bundle.sh > /dev/null

# bundle and sign outside iCloud-synced folders (Desktop, Documents): iCloud tags files
# with Finder info, and codesign refuses to sign anything that carries it
STAGE_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/armagetronad-build.XXXXXX")
STAGE="$STAGE_ROOT/package.noindex"
mkdir -p "$STAGE"
trap 'rm -rf "$STAGE_ROOT"' EXIT
bash desktop/os-x/build_bundle.sh "${STAGE}"
for f in "${STAGE}"/*.zip "${STAGE}"/*.dmg; do
    test -f "$f" || continue
    ditto "$f" "${OUT}/$(basename "$f")"
done
python3 "$SRC/desktop/os-x/install_native_app.py" \
    "$STAGE/Armagetron Advanced.app" --backup "$OUT/previous-installed.zip"
ls -d "/Applications/Armagetron Advanced.app" "${OUT}"/*.dmg "${OUT}"/*.zip
