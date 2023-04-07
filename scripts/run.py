import argparse

from typing import Dict

def run(conf: Dict) -> None:
    pass

def parse_args() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--template",
        type=str,
        default="...",
        help="...",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    run(conf)