#!/usr/bin/env python3
"""Install the one canonical Armagetron app; retain rollback data as a ZIP."""
import argparse
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile

DESTINATION = Path('/Applications/Armagetron Advanced.app')
LSREGISTER = Path('/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister')


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def install(source, backup):
    source = source.resolve()
    with (source / 'Contents/Info.plist').open('rb') as f:
        info = plistlib.load(f)
    if info.get('CFBundleIdentifier') != 'org.armagetronad':
        raise ValueError('Source is not the Armagetron Advanced client bundle')
    run('codesign', '--verify', '--deep', '--strict', source)
    if source == DESTINATION:
        return
    subprocess.run([str(LSREGISTER), '-u', str(source)], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        temporary_zip = backup.with_suffix('.pending.zip')
        run('/usr/bin/ditto', '-c', '-k', '--keepParent', DESTINATION, temporary_zip)
        run('/usr/bin/unzip', '-tq', temporary_zip)
        temporary_zip.replace(backup)
    # Never leave another discoverable .app in the project or in a backup folder.
    # The temporary rollback bundle is outside indexing and is removed only after
    # both the archive and the replacement signature have been verified.
    with tempfile.TemporaryDirectory(prefix='armagetron-install-', suffix='.noindex') as directory:
        rollback = Path(directory) / 'previous.app'
        if DESTINATION.exists():
            shutil.move(str(DESTINATION), rollback)
        try:
            shutil.move(str(source), DESTINATION)
            run('codesign', '--verify', '--deep', '--strict', DESTINATION)
        except BaseException:
            if DESTINATION.exists():
                if not source.exists():
                    shutil.move(str(DESTINATION), source)
                else:
                    shutil.move(str(DESTINATION), Path(directory) / 'failed.app')
            if rollback.exists():
                shutil.move(str(rollback), DESTINATION)
            raise
    run(LSREGISTER, '-f', DESTINATION)
    subprocess.run(['/usr/bin/mdimport', str(DESTINATION)], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'Installed: {DESTINATION}')
    print(f'Previous version archive: {backup}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--backup', type=Path, required=True)
    args = parser.parse_args()
    install(args.source, args.backup.resolve())
