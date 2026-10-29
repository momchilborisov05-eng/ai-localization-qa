"""
Command-line interface.

    python -m locqa demo                                  # end-to-end, offline, no API key
    python -m locqa translate FILE... -t bg de [options]  # translate + QA + MQM review
    python -m locqa qa FILE -t de [options]               # QA an already translated file
    python -m locqa feedback FEEDBACK.csv [options]       # post-edit metrics, TM update, glossary candidates
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .demo import SAMPLES_DIR, DemoClient
from .errors import LocQAError
from .feedback import (analyse, glossary_candidates, load_feedback, tm_leverage, update_tm,
                       write_candidates_csv)
from .formats import display_path, load_document
from .glossary import Glossary
from .pipeline import Options, Resources, RunResult, qa_file, translate_file
from .report import build_report, write_html, write_json
from .tm import TranslationMemory


def _read_text(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        raise LocQAError(f"File not found: {p}")
    return p.read_text(encoding="utf-8")


def _make_client(demo: bool):
    if demo:
        return DemoClient()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LocQAError("ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example) or use --demo.")
    import anthropic
    return anthropic.Anthropic()


def _resources(args) -> Resources:
    return Resources(
        glossary=Glossary.load(args.glossary),
        tm=TranslationMemory.load(args.tm, args.source_lang),
        style_guide=_read_text(getattr(args, "style_guide", None)),
        domain_context=_read_text(getattr(args, "context", None)),
    )


def _print_run(run: RunResult) -> None:
    s = run.summary()
    if s["existing"]:
        source = f"{s['existing']} existing translations checked"
    else:
        source = (f"TM reused {s['tm_reused']}, MT {s['mt_with_fuzzy'] + s['mt_new']} "
                  f"({s['mt_with_fuzzy']} with fuzzy ref)")
    line = (f"  {Path(run.input_file).name} -> {run.target_lang}: {s['segments']} segments | {source}"
            f" | QA {_plural(s['qa_errors'], 'error')}, {_plural(s['qa_warnings'], 'warning')}")
    if s["repaired"]:
        line += f" | {s['repaired']} repaired"
    if s["mqm"]:
        m = s["mqm"]
        line += f" | MQM {m['score']:.1f} {'PASS' if m['passed'] else 'FAIL'}"
    print(line)
    if s["unwritable"]:
        print(f"    ! target left empty (broken tags): {', '.join(s['unwritable'])}")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _write_reports(out: Path, report: dict, name: str = "report") -> None:
    write_json(report, out / f"{name}.json")
    write_html(report, out / f"{name}.html")
    print(f"\nReport: {out / (name + '.html')}\n        {out / (name + '.json')}")


def _exit_code(runs: list[RunResult], fail_on_errors: bool) -> int:
    if not fail_on_errors:
        return 0
    for run in runs:
        s = run.summary()
        if s["qa_errors"] or (s["mqm"] and not s["mqm"]["passed"]):
            return 2
    return 0


# ---------- commands ----------

def cmd_translate(args) -> int:
    client = _make_client(args.demo)
    resources = _resources(args)
    options = Options(batch_size=args.batch_size, fuzzy_min=args.fuzzy_min, review=not args.no_review,
                      mqm_threshold=args.mqm_threshold, source_lang=args.source_lang)
    out = Path(args.out)
    runs = []
    print(f"Translating {len(args.files)} file(s) into {', '.join(args.target_lang)}"
          f"{' (demo mode: canned responses, no API calls)' if args.demo else ''}")
    for file in args.files:
        for lang in args.target_lang:
            run = translate_file(Path(file), lang, out, resources, client, options, effort=args.effort)
            _print_run(run)
            runs.append(run)
    _write_reports(out, build_report(runs, demo=args.demo, settings=_settings(args)))
    return _exit_code(runs, args.fail_on_errors)


def cmd_qa(args) -> int:
    client = _make_client(args.demo) if args.review else None
    resources = _resources(args)
    options = Options(review=args.review, mqm_threshold=args.mqm_threshold, source_lang=args.source_lang)
    run = qa_file(Path(args.file), args.target_lang, resources, client, options,
                  target_path=Path(args.target) if args.target else None, effort=args.effort)
    _print_run(run)
    _write_reports(Path(args.out), build_report([run], demo=bool(args.review and args.demo), settings=_settings(args)),
                   name="qa_report")
    return _exit_code([run], args.fail_on_errors)


def run_feedback(feedback_csv: Path, tm_path: Path | None, glossary_path: Path | None, out: Path,
                 source_lang: str = "en", source_file: Path | None = None) -> dict:
    rows = load_feedback(feedback_csv)
    glossary = Glossary.load(glossary_path)
    tm = TranslationMemory.load(tm_path, source_lang)
    sources = [s.source for s in load_document(source_file, source_lang).segments] if source_file else []
    languages = sorted({r.target_lang for r in rows})
    before = {lang: tm_leverage(sources, tm, lang) for lang in languages} if sources else {}

    tm_counts = update_tm(rows, tm, source_lang)
    tm_out = out / "tm_updated.tmx"
    tm.save_tmx(tm_out)
    candidates = glossary_candidates(rows, glossary)
    write_candidates_csv(candidates, out / "glossary_candidates.csv")

    leverage = {}
    for lang in before:
        leverage[lang] = {"segments": len(sources), "before": before[lang], "after": tm_leverage(sources, tm, lang)}
    return {
        "feedback_file": display_path(feedback_csv),
        "metrics": analyse(rows, glossary),
        "tm_update": tm_counts,
        "tm_size": len(tm),
        "tm_output": display_path(tm_out),
        "glossary_candidates": [c.to_dict() for c in candidates],
        "glossary_candidates_file": display_path(out / "glossary_candidates.csv"),
        "tm_leverage": leverage,
    }


def _print_feedback(fb: dict) -> None:
    o = fb["metrics"]["overall"]
    print(f"  {o['segments']} post-edited segments | unchanged {o['unchanged_pct']}% | "
          f"avg edit distance {o['avg_normalized_edit_distance']:.3f} | avg edit rate {o['avg_edit_rate']:.3f}")
    for lang, a in fb["metrics"]["languages"].items():
        qa = fb["metrics"]["qa_before_after"][lang]
        line = (f"  {lang}: {a['segments']} segments, effort {a['effort']} | "
                f"QA errors MT {qa['mt_errors']} -> post-edit {qa['post_edit_errors']}")
        if lang in fb["tm_leverage"]:
            lev = fb["tm_leverage"][lang]
            line += f" | TM exact matches {lev['before']['exact']} -> {lev['after']['exact']} of {lev['segments']}"
        print(line)
    t = fb["tm_update"]
    print(f"  TM: +{t['added']} added, {t['updated']} updated -> {fb['tm_output']}")
    for c in fb["glossary_candidates"]:
        status = (f"existing term '{c['glossary_term']}' not followed by MT" if c["status"] == "glossary-not-followed"
                  else "new candidate")
        print(f"  glossary [{c['target_lang']}] '{c['mt_term']}' -> '{c['post_edit_term']}' ({c['occurrences']}x, {status})")


def cmd_feedback(args) -> int:
    out = Path(args.out)
    print("Analysing linguist feedback")
    fb = run_feedback(Path(args.file), Path(args.tm) if args.tm else None,
                      Path(args.glossary) if args.glossary else None, out, args.source_lang,
                      Path(args.source_file) if args.source_file else None)
    _print_feedback(fb)
    write_json(fb, out / "feedback_report.json")
    write_html(build_report([], feedback=fb), out / "feedback_report.html")
    print(f"\nReport: {out / 'feedback_report.html'}")
    return 0


def cmd_demo(args) -> int:
    out = Path(args.out)
    s = SAMPLES_DIR
    print("AI Localization QA - demo mode")
    print("Translations and review findings come from canned responses in demo/canned/ (no API key, no API calls).\n")
    resources = Resources(
        glossary=Glossary.load(s / "glossary.csv"),
        tm=TranslationMemory.load(s / "tm.tmx"),
        style_guide=(s / "style_guide.md").read_text(encoding="utf-8"),
        domain_context=(s / "domain_context.md").read_text(encoding="utf-8"),
    )
    client = DemoClient()
    runs = []
    print("1) Translate + QA + MQM review")
    for file in (s / "product_copy.en.xlf", s / "ui_strings.en.json"):
        for lang in ("bg", "de"):
            run = translate_file(file, lang, out, resources, client, Options())
            _print_run(run)
            runs.append(run)
    print(f"   {len(client.requests)} requests answered by the demo client")

    print("\n2) Linguist feedback loop")
    fb = run_feedback(s / "linguist_feedback.csv", s / "tm.tmx", s / "glossary.csv", out,
                      source_file=s / "product_copy.en.xlf")
    _print_feedback(fb)
    write_json(fb, out / "feedback_report.json")

    _write_reports(out, build_report(runs, feedback=fb, demo=True, settings={"mode": "demo"}))
    print(f"Translated files are in {out}")
    return 0


def _settings(args) -> dict:
    keys = ("target_lang", "glossary", "tm", "style_guide", "context", "batch_size", "fuzzy_min", "mqm_threshold", "effort")
    return {k: getattr(args, k) for k in keys if hasattr(args, k)}


# ---------- argument parsing ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="locqa", description="AI translation with TM, glossary, automatic QA and MQM review.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, out_default):
        sp.add_argument("--glossary", help="glossary CSV (term, <lang> columns, do_not_translate)")
        sp.add_argument("--tm", help="translation memory (.tmx or .csv)")
        sp.add_argument("--source-lang", default="en", help="source language (default: from the file, else en)")
        sp.add_argument("--out", default=out_default, help=f"output folder (default: {out_default})")

    t = sub.add_parser("translate", help="translate files with Claude, then run QA and MQM review")
    t.add_argument("files", nargs="+", help="XLIFF 1.2/2.0 (.xlf, .xliff, .mxliff) or JSON files")
    t.add_argument("-t", "--target-lang", nargs="+", required=True, help="target language codes, e.g. bg de")
    common(t, "outputs")
    t.add_argument("--style-guide", help="style guide (Markdown or text)")
    t.add_argument("--context", help="domain/client context (Markdown or text)")
    t.add_argument("--batch-size", type=int, default=15, help="segments per request (default: 15)")
    t.add_argument("--fuzzy-min", type=int, default=75, help="minimum TM match %% sent as reference (default: 75)")
    t.add_argument("--mqm-threshold", type=float, default=95.0, help="MQM pass threshold (default: 95)")
    t.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    t.add_argument("--no-review", action="store_true", help="skip the LLM MQM review")
    t.add_argument("--fail-on-errors", action="store_true", help="exit with code 2 on QA errors or MQM fail (for CI)")
    t.add_argument("--demo", action="store_true", help="use canned responses (bundled samples only)")
    t.set_defaults(func=cmd_translate)

    q = sub.add_parser("qa", help="QA an already translated file (no translation)")
    q.add_argument("file", help="bilingual XLIFF, or the source JSON (with --target)")
    q.add_argument("-t", "--target-lang", help="target language (default: from the XLIFF)")
    q.add_argument("--target", help="translated JSON file, when FILE is a source JSON")
    common(q, "outputs")
    q.add_argument("--review", action="store_true", help="also run the LLM MQM review (needs an API key)")
    q.add_argument("--style-guide", help="style guide for the review")
    q.add_argument("--mqm-threshold", type=float, default=95.0)
    q.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    q.add_argument("--fail-on-errors", action="store_true")
    q.add_argument("--demo", action="store_true", help=argparse.SUPPRESS)
    q.set_defaults(func=cmd_qa)

    f = sub.add_parser("feedback", help="analyse linguist post-edits and update TM / glossary candidates")
    f.add_argument("file", help="CSV: segment_id, target_lang, source, mt, post_edit, category, comment")
    common(f, "outputs")
    f.add_argument("--source-file", help="source file to measure TM leverage before/after the update")
    f.set_defaults(func=cmd_feedback)

    d = sub.add_parser("demo", help="run the full workflow on the sample files, offline")
    d.add_argument("--out", default="outputs/demo", help="output folder (default: outputs/demo)")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LocQAError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        if not type(e).__module__.startswith("anthropic"):
            raise
        print(f"Error: {_api_error_message(e)}", file=sys.stderr)
        return 1


def _api_error_message(e: Exception) -> str:
    # anthropic is imported lazily (it is slow to import and not needed for demo/QA runs)
    import anthropic
    if isinstance(e, anthropic.AuthenticationError):
        return "the API key was rejected. Check ANTHROPIC_API_KEY in .env."
    if isinstance(e, anthropic.RateLimitError):
        return "rate limit reached. Wait a minute or use a smaller --batch-size."
    if isinstance(e, anthropic.APIStatusError):
        return f"the Claude API returned {e.status_code}: {e.message}"
    if isinstance(e, anthropic.APIConnectionError):
        return "could not reach the Claude API. Check your internet connection."
    return str(e)
