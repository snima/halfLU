#!/usr/bin/env python3
"""Aggregator for the fused device-side pivot search campaign (Table 7 EXT).

Produces the paper-ready comparison: for every (schedule, family, order,
method), the median wall time of the host path and of the fused path, each
expressed as an overhead relative to the SAME-PATH PP baseline.

Reporting the two overheads side by side is the whole point. The frozen Table 7
column is "overhead of rule X over host PP"; the new column is "overhead of rule
X over fused PP". Because the fused path also accelerates PP, the second column
is the honest one: it isolates the cost of the pivot RULE from the cost of the
HARNESS. Rules whose overhead survives (RP, ScPP) pay a genuine algorithmic
price; rules whose overhead collapses (CP, ScaP) were paying for host round
trips.

Outputs, in --output:
    search_cost_raw.csv          every run, tidied
    search_cost_summary.csv      per (schedule, family, n, method) medians
    search_cost_table.md         paper-ready markdown table
    search_cost_report.md        headline findings and equivalence audit
    search_cost_summary.json     machine-readable summary

Usage:
    python3 analyze_search_cost.py ./campaign_root --output ./analysis
"""

import argparse
import collections
import csv
import json
import os
import statistics
import sys

METHOD_ORDER = ["PP", "DP", "GP", "ScaP", "RP", "CP", "ScPP"]
COMPLETED = "completed"


def load_rows(root):
    rows = []
    for directory, _unused, files in os.walk(root):
        for name in files:
            if not name.endswith(".csv"):
                continue
            path = os.path.join(directory, name)
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or "search_mode" not in reader.fieldnames:
                    continue
                for row in reader:
                    row["_source"] = os.path.relpath(path, root)
                    rows.append(row)
    return rows


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", default="factor_wall_ms",
                        choices=["factor_wall_ms", "factor_cuda_ms"])
    arguments = parser.parse_args()

    rows = load_rows(arguments.root)
    if not rows:
        print("error: no CSV rows with a search_mode column under " + arguments.root,
              file=sys.stderr)
        return 2
    os.makedirs(arguments.output, exist_ok=True)

    with open(os.path.join(arguments.output, "search_cost_raw.csv"), "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # (schedule, family, n, method, search) -> [times]
    times = collections.defaultdict(list)
    statuses = collections.defaultdict(set)
    digests = collections.defaultdict(set)
    counters = collections.defaultdict(lambda: collections.defaultdict(int))
    for row in rows:
        key = (row["schedule"], row["family"], int(row["n"]), row["method"],
               row["search_mode"])
        statuses[key].add(row["status"])
        if row["status"] == COMPLETED:
            times[key].append(to_float(row[arguments.metric]))
        digests[(row["schedule"], row["family"], int(row["n"]), row["method"],
                 int(row["seed"]))].add((row["search_mode"], row["pivot_digest"]))
        for field in ("dp_lookahead_accepts", "dp_fallbacks", "gp_near_ties",
                      "gp_second_choices", "scap_current", "scap_middle",
                      "scap_last", "rp_iterations", "rp_failures", "row_swaps",
                      "column_swaps"):
            if field in row:
                counters[(row["method"], row["search_mode"])][field] += int(row[field] or 0)

    # Equivalence audit: for each matched cell+seed, host and fused digests must agree.
    mismatched = []
    for key, entries in sorted(digests.items()):
        by_mode = dict(entries)
        if "host" in by_mode and "fused_device" in by_mode:
            if by_mode["host"] != by_mode["fused_device"]:
                mismatched.append((key, by_mode["host"], by_mode["fused_device"]))

    summary_rows = []
    cells = sorted({(k[0], k[1], k[2]) for k in times})
    for schedule, family, n in cells:
        baseline = {}
        for search in ("host", "fused_device"):
            values = times.get((schedule, family, n, "PP", search), [])
            baseline[search] = statistics.median(values) if values else None
        for method in METHOD_ORDER:
            record = {
                "schedule": schedule, "family": family, "n": n, "method": method,
            }
            for search in ("host", "fused_device"):
                values = times.get((schedule, family, n, method, search), [])
                key = (schedule, family, n, method, search)
                median = statistics.median(values) if values else None
                record[search + "_ms"] = round(median, 3) if median else None
                record[search + "_runs"] = len(values)
                record[search + "_status"] = ";".join(sorted(statuses.get(key, {"not_attempted"})))
                base = baseline[search]
                record[search + "_overhead_pct"] = (
                    round(100.0 * (median / base - 1.0), 2)
                    if median and base else None)
            if record["host_ms"] and record["fused_device_ms"]:
                record["speedup"] = round(record["host_ms"] / record["fused_device_ms"], 3)
            else:
                record["speedup"] = None
            summary_rows.append(record)

    fields = list(summary_rows[0].keys())
    with open(os.path.join(arguments.output, "search_cost_summary.csv"), "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Paper-ready markdown table.
    lines = ["# Table 7 EXT -- pivot search cost, host vs fused device", "",
             "Metric: median `{}`. Overhead is relative to the PP cell of the".format(arguments.metric),
             "SAME search path, so each column is internally schedule-matched.", ""]
    for schedule, family in sorted({(r["schedule"], r["family"]) for r in summary_rows}):
        lines += ["## {} / {}".format(schedule, family), "",
                  "| n | method | host ms | host ovh | fused ms | fused ovh | speedup |",
                  "|---:|:--|---:|---:|---:|---:|---:|"]
        for record in summary_rows:
            if record["schedule"] != schedule or record["family"] != family:
                continue

            def show(value, suffix=""):
                return "n/a" if value is None else "{}{}".format(value, suffix)

            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                record["n"], record["method"],
                show(record["host_ms"]),
                show(record["host_overhead_pct"], "%"),
                show(record["fused_device_ms"]),
                show(record["fused_device_overhead_pct"], "%"),
                show(record["speedup"], "x")))
        lines.append("")
    with open(os.path.join(arguments.output, "search_cost_table.md"), "w") as handle:
        handle.write("\n".join(lines))

    # Headline report.
    largest = max(r["n"] for r in summary_rows)
    report = ["# Table 7 EXT -- findings", "",
              "Rows: {}. Metric: {}. Largest order: {}.".format(
                  len(rows), arguments.metric, largest), "",
              "## Equivalence audit", ""]
    if mismatched:
        report.append("**FAILED**: {} matched cells disagree on the pivot digest. "
                      "The fused timings do NOT measure the same algorithm and must "
                      "not be published.".format(len(mismatched)))
        for key, host_digest, fused_digest in mismatched[:20]:
            report.append("- {}: host={} fused={}".format(key, host_digest, fused_digest))
    else:
        matched = sum(1 for _key, entries in digests.items()
                      if {mode for mode, _d in entries} == {"host", "fused_device"})
        report.append("PASSED: {} matched (schedule, family, n, method, seed) cells "
                      "have identical pivot digests under both search paths.".format(matched))
    report += ["", "## Overhead collapse", "",
               "| schedule | family | n | method | host ovh | fused ovh |",
               "|:--|:--|---:|:--|---:|---:|"]
    for record in summary_rows:
        host_overhead = record["host_overhead_pct"]
        fused_overhead = record["fused_device_overhead_pct"]
        if host_overhead is None or fused_overhead is None:
            continue
        if host_overhead - fused_overhead > 10.0:
            report.append("| {} | {} | {} | {} | {}% | {}% |".format(
                record["schedule"], record["family"], record["n"], record["method"],
                host_overhead, fused_overhead))
    report += ["", "## Activation counters by search path", "",
               "These must match between paths for the same grid; they are the",
               "evidence that each rule is active rather than an alias for PP.", "",
               "| method | search | DP acc | DP fb | GP ties | GP 2nd | ScaP c/m/l | RP iter |",
               "|:--|:--|---:|---:|---:|---:|:--|---:|"]
    for (method, search), values in sorted(counters.items()):
        report.append("| {} | {} | {} | {} | {} | {} | {}/{}/{} | {} |".format(
            method, search, values["dp_lookahead_accepts"], values["dp_fallbacks"],
            values["gp_near_ties"], values["gp_second_choices"],
            values["scap_current"], values["scap_middle"], values["scap_last"],
            values["rp_iterations"]))
    with open(os.path.join(arguments.output, "search_cost_report.md"), "w") as handle:
        handle.write("\n".join(report) + "\n")

    with open(os.path.join(arguments.output, "search_cost_summary.json"), "w") as handle:
        json.dump({"metric": arguments.metric,
                   "rows": len(rows),
                   "equivalence_mismatches": len(mismatched),
                   "summary": summary_rows}, handle, indent=2)

    print("raw      : {}/search_cost_raw.csv".format(arguments.output))
    print("summary  : {}/search_cost_summary.csv".format(arguments.output))
    print("table    : {}/search_cost_table.md".format(arguments.output))
    print("report   : {}/search_cost_report.md".format(arguments.output))
    print()
    print("equivalence mismatches: {}".format(len(mismatched)))
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
