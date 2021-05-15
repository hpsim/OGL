#!/usr/bin/env python3
import sys


def extract_iters_from_line(line, pos):
    return float(line.split(",")[pos])


def relative_difference(res1, res2):
    return abs(res1 - res2) / res1


if __name__ == "__main__":

    fn = sys.argv[1]
    with open(fn) as fh:
        lines = fh.readlines()[1:]
        for i in range(int(len(lines) / 2)):
            rel_diff = relative_difference(
                extract_iters_from_line(lines[2 * i], -1),
                extract_iters_from_line(lines[2 * i + 1], -1),
            )
            if rel_diff > 0.05:
                sys.exit(1)
