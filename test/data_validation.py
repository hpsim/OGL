import os
import sys

def test_matrix_presence(workspace: str):
    """This function checks for the pressence of matrix files"""
    mtx_files = []
    for root, folder, files in os.walk(workspace):
        for file in files:
            if file.endswith(".mtx"):
                mtx_files.append(f"{root}/{file}")
    if not mtx_files:
        raise ValueError("no .mtx files found")
        sys.exit(1)
    print(f"PASS found {len(mtx_files)} .mtx files")


def main():
    workspace = sys.argv[1]
    test_matrix_presence(workspace)


if __name__ == "__main__":
    main()


