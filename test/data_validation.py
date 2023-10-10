import os
import sys
import collections

from subprocess import check_output
from logging import info


def find_mtx_files(workspace: str) -> tuple:
    mtx_files = []
    for root, folder, files in os.walk(workspace):
        for file in files:
            if file.endswith(".mtx"):
                time = root.split("/")[-1]
                proc = root.split("/")[-2]
                uid = root.split("/")[-4]
                field_name = file.split("_")[0]
                file_path = f"{root}/{file}"
                md5sum = check_output(["md5sum", file_path], text=True).split()[0]

                mtx_files.append(
                    {
                        "time": time,
                        "proc": proc,
                        "field": field_name,
                        "path": file_path,
                        "md5sum": md5sum,
                        "uid": uid,
                    }
                )
    return mtx_files


def test_matrix_presence(workspace: str):
    """This function checks for the presence of matrix files"""
    print("check if matrix files exist")
    mtx_files = find_mtx_files(workspace)
    if not mtx_files:
        raise ValueError("no .mtx files found")
        sys.exit(1)
    print(f"PASS found {len(mtx_files)} .mtx files")


def test_if_matrices_are_unique(workspace):
    """This function checks if matrices are different for different timesteps"""
    mtx_files = find_mtx_files(workspace)
    md5sums = [
        (record["field"], record["proc"], record["md5sum"], record["uid"])
        for record in mtx_files
        if (record["time"] == "0.05") or (record["time"] == "0.5")
    ]
    duplicates = [
        item for item, count in collections.Counter(md5sums).items() if count > 1
    ]

    if len(duplicates) != 0:
        for dup in duplicates:
            for mtx_file in mtx_files:
                if mtx_file["md5sum"] == dup[2]:
                    print(mtx_file)

        raise ValueError("Not all .mtx files are unique")
        sys.exit(1)
    print(f"PASS all tested {len(md5sums)} .mtx files are unique")


def verify_local_matrix(lines, file_path):
    for line in lines:
        row, col, val = line.split(" ")
        val = float(val)
        if row == col:
            if (val > 0.001) or (val < 0.0):
                raise ValueError(
                    f"diag value {row} {col} {val} is out of bounds 0, {file_path}"
                )
                sys.exit(1)
        else:
            if (val > 0) or (val < -0.001):
                raise ValueError(
                    f"off diag {row} {col} {val} is out of bounds 0, {file_path}"
                )
                sys.exit(1)


def test_matrix_value_bounds(workspace):
    """This function checks if matrices are coeffs are within bounds"""
    mtx_files = find_mtx_files(workspace)
    for record in mtx_files:
        file_path = record["path"]
        if not "_local_" in file_path:
            continue
        with open(file_path) as fh:
            verify_local_matrix(fh.readlines()[2:], file_path)
    print(f"PASS all tested .mtx files are within bounds")


def main():
    workspace = sys.argv[1]
    test_matrix_presence(workspace)
    test_if_matrices_are_unique(workspace)
    test_matrix_value_bounds(workspace)


if __name__ == "__main__":
    main()
