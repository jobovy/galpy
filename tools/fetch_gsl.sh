#!/bin/bash
# Download the GSL source tarball, trying several mirrors and verifying its
# SHA-256, for builds that compile GSL from source (the Pyodide wheels).
#
#   tools/fetch_gsl.sh VERSION [SHA256]
#
# Writes gsl-VERSION.tar.gz to the current directory. ftp.gnu.org and its
# ftpmirror.gnu.org redirector can be down together, so independent mirrors
# follow; the checksum makes them interchangeable.
set -euo pipefail
VERSION=${1:?usage: fetch_gsl.sh VERSION [SHA256]}
case "$VERSION" in
    2.8) KNOWN=6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190 ;;
    *) KNOWN= ;;
esac
SHA256=${2:-$KNOWN}
if [ -z "$SHA256" ]; then
    echo "fetch_gsl.sh: no known SHA-256 for GSL $VERSION; pass it as the second argument" >&2
    exit 2
fi
TARBALL=gsl-$VERSION.tar.gz
MIRRORS=(
    https://ftpmirror.gnu.org/gsl
    https://ftp.gnu.org/gnu/gsl
    https://mirrors.kernel.org/gnu/gsl
    https://mirror.csclub.uwaterloo.ca/gnu/gsl
    https://mirrors.ocf.berkeley.edu/gnu/gsl
    https://ftp.fau.de/gnu/gsl
    https://mirrors.dotsrc.org/gnu/gsl
)
for mirror in "${MIRRORS[@]}"; do
    echo "fetch_gsl.sh: trying $mirror"
    # bounded per mirror (curl applies --max-time to EACH attempt): <= 2 x 90 s,
    # and a mirror trickling under 20 kB/s for 20 s is abandoned early
    if curl -fsSL --connect-timeout 20 --max-time 90 --retry 1 \
        --speed-limit 20000 --speed-time 20 \
        -o "$TARBALL" "$mirror/$TARBALL"; then
        if echo "$SHA256  $TARBALL" | sha256sum -c --status; then
            echo "fetch_gsl.sh: $TARBALL from $mirror (SHA-256 verified)"
            exit 0
        fi
        echo "fetch_gsl.sh: SHA-256 mismatch from $mirror" >&2
    fi
    rm -f "$TARBALL"
done
echo "fetch_gsl.sh: could not download $TARBALL from any mirror" >&2
exit 1
