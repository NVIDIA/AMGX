#!/bin/sh
# Regression test: 3D readGeometry() must allocate geo_z (not geo_y twice).
#
# Bug: in the 3D branch of readGeometry() in examples/amgx_capi.c and
# examples/amgx_capi_multi.c, `*geo_y` was allocated a second time instead
# of `*geo_z`, so the subsequent fscanf wrote through the NULL *geo_z
# pointer -> segmentation fault on any 3D geometry input file.
#
# The test extracts readGeometry() verbatim from each example source,
# compiles it into a small harness with gcc, and runs it against a 3D
# geometry file. Pre-fix code segfaults (exit 139); fixed code parses and
# returns the correct z values.

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname -- "$SCRIPT_DIR")
TMPDIR_TEST=$(mktemp -d)
trap 'rm -rf "$TMPDIR_TEST"' EXIT

cat > "$TMPDIR_TEST/harness.c" <<'EOF'
#include <stdio.h>
#include <stdlib.h>

static void errAndExit(const char *err)
{
    fprintf(stderr, "%s", err);
    exit(1);
}

/* readGeometry() extracted verbatim from the example source under test */
#include "readGeometry.inc"

int main(void)
{
    /* 3 rows of 3D geometry */
    FILE *f = fopen("geo3.txt", "w");
    if (!f) return 10;
    fprintf(f, "3 3\n1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n");
    fclose(f);

    double *gx = NULL, *gy = NULL, *gz = NULL;
    int dim = 0, n = 0;
    readGeometry("geo3.txt", &gx, &gy, &gz, &dim, &n);

    if (dim != 3 || n != 3) return 2;
    if (gz == NULL) return 3;                       /* geo_z never allocated */
    if (gz[0] != 3.0 || gz[1] != 6.0 || gz[2] != 9.0) return 4;
    if (gy[0] != 2.0 || gy[1] != 5.0 || gy[2] != 8.0) return 5;

    free(gx);
    free(gy);
    free(gz);
    return 0;
}
EOF

status=0
for src in examples/amgx_capi.c examples/amgx_capi_multi.c; do
    name=$(basename "$src" .c)
    awk '/^void readGeometry/{f=1} f{print} f && /^}/{exit}' \
        "$REPO_ROOT/$src" > "$TMPDIR_TEST/readGeometry.inc"
    if ! grep -q readGeometry "$TMPDIR_TEST/readGeometry.inc"; then
        echo "FAIL: could not extract readGeometry() from $src"
        status=1
        continue
    fi
    if ! gcc -Wall -Wextra -o "$TMPDIR_TEST/harness" "$TMPDIR_TEST/harness.c"; then
        echo "FAIL: harness for $src did not compile"
        status=1
        continue
    fi
    (cd "$TMPDIR_TEST" && ./harness)
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "PASS: $name readGeometry() handles 3D input (geo_z allocated)"
    else
        echo "FAIL: $name readGeometry() exited $rc on 3D input" \
             "(geo_z not allocated -> NULL write/segfault)"
        status=1
    fi
done

exit "$status"
