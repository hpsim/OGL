#!/usr/bin/env python3
import sys


def extract_iters_from_line(line, pos):
    return float(line.split(",")[pos])


def relative_difference(res1, res2):
    return abs(res1 - res2) / res1


if __name__ == "__main__":

    fn = sys.argv[1]
    with open(fn) as fh:
        # TODO the first and second line have to be skipped
        # the starting line should be found automatically
        lines = fh.readlines()[2:]
        position_iters_in_log = -3
        for i in range(int(len(lines) / 2)):
            rel_diff = relative_difference(
                extract_iters_from_line(lines[2 * i], position_iters_in_log),
                extract_iters_from_line(lines[2 * i + 1], position_iters_in_log),
            )
            if rel_diff > 0.05:
                sys.exit(1)
