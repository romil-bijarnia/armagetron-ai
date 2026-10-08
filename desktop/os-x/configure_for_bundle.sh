#!/bin/bash

# configures a native app bundle build against Homebrew for the chip of this Mac
# (Homebrew lives in /opt/homebrew on Apple Silicon and in /usr/local on Intel)
BREW=$(brew --prefix 2>/dev/null)
test -z "${BREW}" && BREW=/usr/local
echo BREW=${BREW}

# prefer protobuf 21.X where Homebrew still has it,
# because 22 introduced the unwanted dependency on abseil
OLD_PROTOBUF="${BREW}/opt/protobuf@21"
if test -d "${OLD_PROTOBUF}"; then
    echo OLD_PROTOBUF=${OLD_PROTOBUF}
    export PKG_CONFIG_PATH=${OLD_PROTOBUF}/lib/pkgconfig:${PKG_CONFIG_PATH}
    export PATH=${OLD_PROTOBUF}/bin:${PATH}
fi

# libxml2 is keg-only in Homebrew; everything else is found under the prefix
export PKG_CONFIG_PATH=${BREW}/opt/libxml2/lib/pkgconfig:${BREW}/lib/pkgconfig:${PKG_CONFIG_PATH}
export CPPFLAGS="-I${BREW}/include ${CPPFLAGS}"
export LDFLAGS="-L${BREW}/lib -headerpad_max_install_names ${LDFLAGS}"

$(dirname $0)/../../configure --disable-restoreold --enable-automakedefaults \
    --disable-useradd --disable-sysinstall --disable-initscripts \
    --disable-uninstall --disable-etc --disable-games \
    --with-boost=${BREW} \
    --prefix=/Contents \
    --bindir=/Contents/MacOS \
    --datadir=/Contents/Resources \
    --libdir=/Contents/Frameworks \
    "$@"
