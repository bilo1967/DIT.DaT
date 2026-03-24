"""ditdat CLI — speaker diarization and transcription pipeline."""

import argparse
import sys

from ditdat import phase0, phase1, phase2, phase2_report, phase3, phase4, phase5, phase6


def build_run_parser(subparsers):
    """Full pipeline subcommand (runs phase0 → phase3, like the old pipeline.py)."""
    p = subparsers.add_parser(
        "diarize",
        help="Split per-speaker audio (phase0 through phase3)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Collect all args from every phase, deduplicating shared ones
    # (--project-dir, --force, etc. are defined once below)
    p.add_argument("--project-dir", required=True, help="Project directory")
    p.add_argument("--force", action="store_true", help="Overwrite existing output without asking")
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    # phase0-specific
    p.add_argument("--input", type=str, required=True, help="Input audio file")
    block_group = p.add_mutually_exclusive_group(required=True)
    block_group.add_argument("--block-duration", type=str, help="Target block duration (e.g. 30m, 1800s)")
    block_group.add_argument("--num-blocks", type=int, help="Number of blocks to split audio into")
    p.add_argument("--sample-duration", type=float, default=60.0, help="Speaker sample duration in seconds [default: 60]")
    p.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
    p.add_argument("--max-speakers", type=int, help="Maximum number of speakers")
    p.add_argument("--num-speakers", type=int, help="Exact number of speakers")
    p.add_argument("--exclusive-mode", action="store_true", help="Use exclusive diarization mode")
    p.add_argument("--cpu", action="store_true", help="Force CPU (no GPU)")
    p.add_argument("--token", type=str, help="Hugging Face Hub token")
    p.add_argument("--residual-threshold", type=int, default=5, help="Residual block threshold [default: 5]")
    p.add_argument("--auto-map", action="store_true", help="Auto-generate speaker mapping using embeddings")

    # phase1-specific
    p.add_argument("--speaker-map", help="Path to speakers_map.txt (default: project-dir/speakers_map.txt)")
    p.add_argument("--output-json", help="Phase 1 output JSON path")

    # phase2-specific
    p.add_argument("--input-json", help="Phase 1 JSON input (default: project-dir/fase1_unified.json)")
    p.add_argument("--min-pause", type=float, help="Merge threshold in seconds [default: from config or 1.5]")
    p.add_argument("--min-duration", type=float, help="Minimum segment duration in seconds [default: from config or 0.5]")
    p.add_argument("--merge-speakers", type=str, default="", help="Merge speakers (e.g. 'A=B,C=D')")
    p.add_argument("--rename-speakers", type=str, default="", help="Rename speakers (e.g. 'A=Host,B=Guest')")
    p.add_argument("--drop-speakers", type=str, default="", help="Drop speakers (e.g. 'A,B')")
    p.add_argument("--verbose", action="store_true", help="Verbose output")

    # phase3-specific
    p.add_argument("--input-wav", help="WAV file path (default: from config or JSON)")
    p.add_argument("--output-dir", help="Output directory (default: project-dir/combined)")
    p.add_argument("--fill-mode", choices=["none", "white", "pink"], default="none",
                    help="Gap fill mode [default: none]")
    p.add_argument("--speaker", help="Process only this speaker")
    p.add_argument("--dump-segments", action="store_true", help="Save individual audio fragments")

    # phase4-specific


    # phase5-specific


    # phase6-specific

    return p


def main():
    parser = argparse.ArgumentParser(
        prog="ditdat",
        description="DIT.DaT — Speaker diarization and transcription pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # Individual phase subcommands
    p0 = subparsers.add_parser("phase0", help="Diarization and block splitting",
                                formatter_class=argparse.RawTextHelpFormatter)
    phase0.add_arguments(p0)

    p1 = subparsers.add_parser("phase1", help="Unify blocks and apply speaker mapping",
                                formatter_class=argparse.RawTextHelpFormatter)
    phase1.add_arguments(p1)

    p2 = subparsers.add_parser("phase2", help="Merge segments and filter short ones",
                                formatter_class=argparse.RawTextHelpFormatter)
    phase2.add_arguments(p2)

    p2r = subparsers.add_parser("phase2-report", help="Generate interactive HTML report",
                                formatter_class=argparse.RawTextHelpFormatter)
    phase2_report.add_arguments(p2r)

    p3 = subparsers.add_parser("phase3", help="Extract per-speaker audio files",
                                formatter_class=argparse.RawTextHelpFormatter)
    phase3.add_arguments(p3)

    p4 = subparsers.add_parser("phase4", help="Whisper transcription",
                                formatter_class=argparse.RawTextHelpFormatter)
    phase4.add_arguments(p4)

    p5 = subparsers.add_parser("phase5", help="Generate SRT and TXT output files",
                                formatter_class=argparse.RawTextHelpFormatter)
    phase5.add_arguments(p5)

    p6 = subparsers.add_parser("phase6", help="Validate and regenerate subtitles after revision",
                                formatter_class=argparse.RawTextHelpFormatter)
    phase6.add_arguments(p6)

    # Full pipeline
    build_run_parser(subparsers)

    args = parser.parse_args()

    phase_map = {
        "phase0": phase0.main,
        "phase1": phase1.main,
        "phase2": phase2.main,
        "phase2-report": phase2_report.main,
        "phase3": phase3.main,
        "phase4": phase4.main,
        "phase5": phase5.main,
        "phase6": phase6.main,
    }

    if args.command in phase_map:
        result = phase_map[args.command](args)
        sys.exit(0 if result is None or result == 0 else result)

    elif args.command == "diarize":
        for fn in [phase0.main, phase1.main, phase2.main, phase3.main]:
            result = fn(args)
            if result and result != 0:
                print(f"Pipeline stopped at {fn.__module__} (exit code {result})")
                sys.exit(result)
        sys.exit(0)


if __name__ == "__main__":
    main()
