import hashlib
import pathlib
from argparse import ArgumentParser

def compute_md5(p):
    with open(p, "rb") as f:
        data = f.read()

    return hashlib.md5(data).hexdigest()


def main(argv=None):
    parser = ArgumentParser()

    parser.add_argument("path1", type=str)
    parser.add_argument("path2", type=str)

    args = parser.parse_args(argv)

    print(args)


    path_1 = pathlib.Path(args.path1)
    path_2 = pathlib.Path(args.path2)
    cwd = pathlib.Path.cwd()

    assert path_1.exists()
    assert path_2.exists()

    if path_1.is_file():
        all_paths_1 = [path_1]
    else:
        all_paths_1 = sorted([p for p in path_1.rglob("*") if p.is_file()])

    if path_2.is_file():
        all_paths_2 = [path_2]
    else:
        all_paths_2 = sorted([p for p in path_2.rglob("*") if p.is_file()])


    # Check length
    assert len(all_paths_1) == len(all_paths_2)
    print("Same number of files")

    # Check name equality
    assert [p.name for p in all_paths_1] == [p.name for p in all_paths_2]
    print("Same file names")

    # Check contents (hash)
    counter_mismatch = 0
    for p_1, p_2 in zip(all_paths_1, all_paths_2):
        matching = compute_md5(p_1) == compute_md5(p_2)
        counter_mismatch += int(not matching)
        res = " == " if matching else " != "
        print(p_1, res, p_2)

    print(f"Found {counter_mismatch} different files")

if __name__ == "__main__":
    main()
