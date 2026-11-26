#!/usr/bin/env python3
"""Parse a rocprofv2 run produced by Tools/rocprof_check.sh.

The measured window is the tail of the trace after the largest idle gap between GPU dispatches -
that gap is the pause between the warm request and the decode-only request. Everything after it is
`--ndec` decoded tokens (plus a few teardown dispatches, which is why the per-token figures are
rounded to 0.1).

Usage: rocprof_parse.py <out-dir> <decode-tokens> [markers.txt]

With markers.txt (lines "tag <monotonic-ns>") the window is the time between req2_start and req2_end,
anchored so the first traced record equals the "spawn" marker. Without it, the window is the tail of
the trace after the largest idle gap, which also catches model load.

"""
import csv
import collections
import glob
import os
import sys


def read_csv(path):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        # rocprofv2 sometimes prefixes rows with a '#' banner; find the header line instead.
        # rocprofv2 prefixes a banner; the header is the first real row (rocprofv2 spells the name
        # column "Name" in kernel traces and "Function" in API traces).
        header = None
        for row in rdr:
            if row and not row[0].startswith("#") and len(row) > 1:
                header = [c.strip().lower() for c in row]
                break
        else:
            return None, []
        out = []
        for row in rdr:
            if not row or row[0].startswith("#") or len(row) < len(header):
                continue
            out.append(dict(zip(header, row)))
        return header, out


def col(header, *cands):
    for c in cands:
        if c in header:
            return c
    return None


def marker_window(records, markers_path):
    """Map shell CLOCK_MONOTONIC markers onto the tracer's clock (same rate, unknown offset)."""
    marks = {}
    for line in open(markers_path):
        parts = line.split()
        if len(parts) == 2:
            marks[parts[0]] = int(parts[1])
    if "spawn" not in marks or "req2_start" not in marks or "req2_end" not in marks:
        return None
    records.sort()
    if not records:
        return None
    offset = records[0][0] - marks["spawn"]
    lo = marks["req2_start"] + offset
    hi = marks["req2_end"] + offset
    sel = [r for r in records if lo <= r[0] <= hi]
    print(f"marker window: offset {offset/1e9:+.3f} s, {len(sel)}/{len(records)} records inside "
          f"{(hi-lo)/1e9:.2f} s")
    return sel


def main(out_dir, ndec, markers_path=None):
    kernels, apis = [], []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.csv"))):
        header, rows = read_csv(path)
        if not rows:
            continue
        name_c = col(header, "name", "function", "api name")
        start_c = col(header, "start (ns)", "start", "start_timestamp")
        dur_c = col(header, "duration (ns)", "duration")
        end_c = col(header, "end (ns)", "end", "end_timestamp")
        if not name_c or not start_c:
            print(f"  (skipping {os.path.basename(path)}: header {header[:6]})")
            continue
        recs = []
        for r in rows:
            try:
                start = int(r[start_c])
            except ValueError:
                continue
            dur = 0
            if dur_c:
                try:
                    dur = int(r[dur_c])
                except ValueError:
                    dur = 0
            elif end_c:
                try:
                    dur = max(0, int(r[end_c]) - start)
                except ValueError:
                    dur = 0
            recs.append((start, dur, r[name_c]))
        is_kernel = "kernel" in os.path.basename(path).lower() or any(
            c in header for c in ("kernel grid", "grid", "workgroup")
        )
        (kernels if is_kernel else apis).extend(recs)
        print(f"  {os.path.basename(path)}: {len(recs)} records ({'kernel' if is_kernel else 'api'})")

    if not kernels and apis:
        # API-only trace: treat the HIP calls themselves as the timeline.
        kernels, apis = apis, []
    if not kernels:
        print("no kernel-trace or API-trace records found")
        return 1

    kernels.sort()
    win = None
    if markers_path and os.path.exists(markers_path):
        win = marker_window(kernels, markers_path)
    if win is None:
        # Largest gap between consecutive dispatch starts = the idle window between requests.
        best_gap, best_i = 0, 0
        for i in range(1, len(kernels)):
            gap = kernels[i][0] - (kernels[i - 1][0] + kernels[i - 1][1])
            if gap > best_gap:
                best_gap, best_i = gap, i
        win = [k for k in kernels if k[0] >= kernels[best_i][0]]
    dur = sum(k[1] for k in win)
    span = win[-1][0] + win[-1][1] - t0
    print(f"\nlargest idle gap {best_gap/1e6:.0f} ms at record {best_i}/{len(kernels)}; "
          f"measuring the {len(win)} dispatches after it ({span/1e6:.0f} ms span, "
          f"{dur/1e6:.1f} ms on-device)")
    print(f"\nkernels per decoded token over {ndec} tokens: {len(win)/ndec:.1f}")

    by_name = collections.Counter(n for _, _, n in win)
    dur_by_name = collections.Counter()
    for _, d, n in win:
        dur_by_name[n] += d
    print(f"\ndistinct kernels: {len(by_name)}   (top 25 by count)")
    print(f"{'count/tok':>10}  {'ms/tok':>7}  name")
    for n, c in by_name.most_common(25):
        print(f"{c/ndec:>10.1f}  {dur_by_name[n]/1e6/ndec:>7.2f}  {n[:110]}")
    print(f"\nsum of on-device kernel time: {dur/1e6/ndec:.2f} ms/token "
          f"({dur/span*100:.1f}% device busy inside the window)")

    if apis:
        apis.sort()
        aw = [a for a in apis if a[0] >= t0]
        cnt = collections.Counter(n for _, _, n in aw)
        print(f"\nHIP API calls after the same marker ({len(aw)}):")
        for n, c in cnt.most_common(12):
            print(f"{c/ndec:>10.1f}/tok  {n}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) == 4 else None))
