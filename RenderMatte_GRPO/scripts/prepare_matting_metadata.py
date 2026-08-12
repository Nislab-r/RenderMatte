import argparse
import json
import os


def resolve_path(root, path, absolute):
    if os.path.isabs(path):
        return path
    joined = os.path.join(root, path) if root else path
    return os.path.abspath(joined) if absolute else joined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--root", default="")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--trimap-index", type=int, default=1)
    parser.add_argument("--alpha-index", type=int, default=-1)
    parser.add_argument(
        "--prompt",
        default="Transform to matting map while maintaining original composition",
    )
    parser.add_argument("--relative", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.split}_metadata.jsonl")
    absolute = not args.relative

    count = 0
    with open(args.split_file, "r", encoding="utf-8") as src, open(
        output_path, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            item = {
                "prompt": args.prompt,
                "image": resolve_path(args.root, parts[args.image_index], absolute),
                "trimap": resolve_path(args.root, parts[args.trimap_index], absolute),
                "alpha": resolve_path(args.root, parts[args.alpha_index], absolute),
            }
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
