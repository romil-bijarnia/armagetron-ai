#!/bin/bash

# One command from a fresh checkout to a native macOS app for the chip of this Mac:
# configures against Homebrew, compiles, and packs the bundle with its libraries inside.
# Result: build-macos/Armagetron Advanced.app, with a .dmg and .zip beside it.
#
# usage: desktop/os-x/build_native_app.sh [build directory, default build-macos]
# needs: brew install autoconf automake boost dylibbundler freetype ftgl glew libpng
#        libxml2 pkgconf protobuf sdl2 sdl2_image sdl2_mixer

set -e

SRC=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-${SRC}/build-macos}

test -x "${SRC}/configure" || (cd "${SRC}" && ./bootstrap.sh)

mkdir -p "${OUT}"
cd "${OUT}"
test -r Makefile || "${SRC}/desktop/os-x/configure_for_bundle.sh"
make -j$(sysctl -n hw.ncpu)
./config.status desktop/os-x/build_bundle.sh > /dev/null

# bundle and sign outside iCloud-synced folders (Desktop, Documents): iCloud tags files
# with Finder info, and codesign refuses to sign anything that carries it
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/armagetronad-app.XXXXXX")
bash desktop/os-x/build_bundle.sh "${STAGE}"
for f in "${STAGE}"/*; do
    rm -rf "${OUT}/$(basename "$f")"
    ditto "$f" "${OUT}/$(basename "$f")"
done
rm -rf "${STAGE}"
ls -d "${OUT}"/*.app "${OUT}"/*.dmg "${OUT}"/*.zip
