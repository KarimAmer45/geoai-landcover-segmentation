"""Command-line interface for both project tracks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geoai", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="download a small basemap GeoTIFF")
    fetch.add_argument("--bbox", nargs=4, required=True, type=float, metavar=("W", "S", "E", "N"))
    fetch.add_argument("--out", default="data/sample/tile.tif")
    fetch.add_argument("--zoom", type=int, default=17)
    fetch.add_argument("--source", default="Satellite")

    segment = subparsers.add_parser("segment", help="run text-prompt segmentation")
    segment.add_argument("--image", default="data/sample/tile.tif")
    segment.add_argument("--prompt", default="trees")
    segment.add_argument("--output-dir", default="outputs")
    segment.add_argument("--model-type", default="sam2-hiera-tiny")
    segment.add_argument("--box-threshold", type=float, default=0.24)
    segment.add_argument("--text-threshold", type=float, default=0.24)
    segment.add_argument("--min-area-ha", type=float, default=0.0)

    embed = subparsers.add_parser("clay-embed", help="generate Clay v1.5 embeddings")
    embed.add_argument("--input", required=True, help="NPZ with chips/timestamps/wavelengths")
    embed.add_argument("--checkpoint", required=True)
    embed.add_argument("--output", default="outputs/clay/embeddings.npz")
    embed.add_argument("--device", default="cpu")

    classify = subparsers.add_parser("clay-classify", help="fit and evaluate a light classifier")
    classify.add_argument("--embeddings", required=True)
    classify.add_argument("--output-dir", default="outputs/clay")
    classify.add_argument("--test-size", type=float, default=0.25)
    classify.add_argument("--bootstrap-iterations", type=int, default=1_000)
    classify.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fetch":
        from .data import fetch_tile

        result = fetch_tile(args.bbox, args.out, zoom=args.zoom, source=args.source)
        print(result)
    elif args.command == "segment":
        from .samgeo_segment import segment_text

        result = segment_text(
            args.image,
            args.prompt,
            args.output_dir,
            model_type=args.model_type,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            min_area_ha=args.min_area_ha,
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif args.command == "clay-embed":
        from .clay_embed import embed_npz

        print(embed_npz(args.input, args.output, args.checkpoint, device=args.device))
    elif args.command == "clay-classify":
        from .clay_embed import run_downstream_experiment

        report = run_downstream_experiment(
            args.embeddings,
            args.output_dir,
            test_size=args.test_size,
            random_state=args.seed,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print(json.dumps(report, indent=2))
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
