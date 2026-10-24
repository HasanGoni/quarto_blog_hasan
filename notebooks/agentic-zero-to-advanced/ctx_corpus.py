"""Context-engineering post: build the "QC archive" the retrieval agent searches.

Everything here is real data from earlier parts of the series, laid out as files:

    out/ctx_archive/
      exports/inspection_log.csv      the full 4,000-row dataset table (the "big export")
      reports/test_00000.md ...       the 200 per-image reports Part 7's pipeline actually wrote
      policy/policy-2026-09-12.md     PLANTED stale policy (I wrote it for this experiment, see the post)
      policy/policy-2026-10-17.md     the current policy: the constants in pipeline_graph.py

Only the policy/ pair is constructed. The old file exists to test whether the agent reads a
superseded document, and the post says so.
"""
import csv
import json
import shutil
from pathlib import Path

import void_data as vd
from pipeline_graph import BAND_MIN_CONTRAST, CONF_MIN, MAX_KEPT, MIN_AREA_PX, SEVERITY_BANDS

HERE = Path(__file__).parent
ARCHIVE = HERE / "out" / "ctx_archive"
OLD_MIN_AREA = 35  # planted: the superseded policy asked for a larger minimum box


def report_markdown(r: dict) -> str:
    rep = r["report"]
    lines = [f"# Report {r['image']}", "",
             f"verdict: {rep['verdict']}",
             f"candidates_examined: {r['n_candidates']}",
             f"candidates_kept: {r['kept']}",
             f"reason: {rep['reason']}", ""]
    for k in rep["kept"]:
        lines.append(f"- kept candidate {k['id']}: severity {k['severity']}, confidence {k['confidence']}, "
                     f"uncertain {str(k['uncertain']).lower()}, contrast {k['contrast']}, area_px {k['area_px']}")
    for k in rep["rejected"]:
        lines.append(f"- rejected candidate {k['id']} (by {k['by']})")
    lines += ["", "## Note", rep["summary"], ""]
    return "\n".join(lines)


def policy_markdown(date: str, min_area: int, status: str, extra: str = "") -> str:
    return f"""# Void reporting policy ({date})

status: {status}

- Minimum reportable void area: {min_area} px. Smaller boxes are not reported.
- Candidates with a classifier confidence below {CONF_MIN} are treated as uncertain.
- More than {MAX_KEPT} surviving candidates on one image means the detector is confused: escalate to a human.
- Severity bands by pixel area: under {SEVERITY_BANDS[0][0]} px minor, under {SEVERITY_BANDS[1][0]} px major, otherwise critical.
- The `band` classifier needs a contrast of at least {BAND_MIN_CONTRAST}.
{extra}"""


def build() -> None:
    if ARCHIVE.exists():
        shutil.rmtree(ARCHIVE)
    for sub in ("exports", "reports", "policy"):
        (ARCHIVE / sub).mkdir(parents=True)
    shutil.copy(vd.DATA_DIR / "metadata.csv", ARCHIVE / "exports" / "inspection_log.csv")
    results = json.loads((HERE / "out" / "pipeline_results.json").read_text())["variants"]["llm"][0]["results"]
    for r in results:
        (ARCHIVE / "reports" / f"{r['image']}.md").write_text(report_markdown(r))
    (ARCHIVE / "policy" / "policy-2026-09-12.md").write_text(
        policy_markdown("2026-09-12", OLD_MIN_AREA, "SUPERSEDED by policy-2026-10-17"))
    (ARCHIVE / "policy" / "policy-2026-10-17.md").write_text(
        policy_markdown("2026-10-17", MIN_AREA_PX, "current"))
    n_rows = sum(1 for _ in csv.reader(open(ARCHIVE / "exports" / "inspection_log.csv"))) - 1
    print(f"archive at {ARCHIVE}\n  {n_rows} rows in inspection_log.csv, {len(results)} reports, 2 policy files")


if __name__ == "__main__":
    build()
