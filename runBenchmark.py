#!/usr/bin/python
"""
    run ogl benchmarks

    Usage:
        runBenchmark.py [options]

    Options:
        -h --help           Show this screen
        -v --version        Print version and exit
        --folder=<folder>   Target folder  [default: Test].
        --report=<filename> Target file to store stats [default: report.csv].
        --of (True|False)   Generate default of cases [default: False].
        --ref (True|False)  Generate ref cases [default: False].
        --cuda (True|False) Generate cuda cases [default: False].
        --omp (True|False)  Generate omp cases [default: False].
        --clean (True|False) Remove existing cases [default: False].
        --large-cases (True|False) Include large cases [default: False].

"""
from docopt import docopt
from subprocess import check_output
from pathlib import Path
import os
import shutil
import datetime


class Results():
    def __init__(self, fn):
        self.fn = Path(fn)
        self.columns = [
            "domain", "executor", "solver", "number_of_iterations",
            "resolution", "run_time"
        ]
        self.current_col_vals = []
        self.report_handle = open(self.fn, "a+")

    def set_case(self, domain, executor, solver, number_of_iterations,
                 resolution):
        self.current_col_vals = [
            domain, executor, solver, number_of_iterations, resolution
        ]

    def add(self, run):

        outp = self.current_col_vals + [run]
        outps = ",".join(map(str, outp))
        print(outps)
        self.report_handle.write(outps + "\n")

    def close_file(self):
        close(self.report_handle)


def sed(fn, in_reg_exp, out_reg_exp, inline=True):
    """ wrapper around sed """
    ret = check_output(
        ["sed", "-i", "s/" + in_reg_exp + "/" + out_reg_exp + "/g", fn])


def clean_block_from_file(fn, block_start, block_end, replace):
    with open(fn, "r") as f:
        lines = f.readlines()
    with open(fn, "w") as f:
        skip = False
        for line in lines:
            if block_start in line:
                skip = True
            if skip == True and block_end in line:
                skip = False
                f.write(replace)
            if not skip:
                f.write(line)


def set_cells(blockMeshDict, old_cells, new_cells):
    """ """
    sed(blockMeshDict, old_cells, new_cells)


def add_libOGL_so(controlDict):
    with open(controlDict, "a") as ctrlDict_handle:
        ctrlDict_handle.write('libs ("libOGL.so");')


def set_end_time(controlDict, endTime):
    sed(controlDict, "endTime[ ]*[0-9.]*", "endTime {}".format(endTime))


def set_deltaT(controlDict, deltaT):
    sed(controlDict, "deltaT[ ]*[0-9.]*", "deltaT {}".format(deltaT))


def clear_solver_settings(fvSolution):
    #sed(fvSolution, "p\\n[ ]*[{][^}]*[}]", "p{}")
    clean_block_from_file(fvSolution, "   p\n", "  }\n", "p{}\n")


def ensure_path(path):
    print("creating", path)
    check_output(["mkdir", "-p", path])


class Case:
    def __init__(self, test_base, of_solver, solver, executor, base_case,
                 resolution, results):
        self.test_base = test_base
        self.fields = "p"
        self.resolution = resolution
        self.of_solver = of_solver
        self.executor = executor
        self.solver = solver
        self.iterations = 1000
        self.results_accumulator = results
        self.results_accumulator.set_case(domain=executor[0],
                                          executor=executor[2],
                                          solver=solver,
                                          number_of_iterations=self.iterations,
                                          resolution=resolution)

        ensure_path(self.path)
        if base_case:
            self.copy_base(base_case, self.path)
        self.set_matrix_solver("boxTurb16")

    @property
    def path(self):
        return Path(
            self.test_base) / Path(self.executor[0] + self.executor[2]) / str(
                self.resolution) / self.fields

    def copy_base(self, src, dst):
        print("copying base case", src, dst)
        check_output(["cp", "-r", src, dst])

    def set_matrix_solver(self, case):
        fn = self.path / case / "system" / "fvSolution"
        matrix_solver = self.executor[1] + self.solver
        solver_str = "p{" + "solver {};tolerance 1e-06; relTol 0.0;smoother none;preconditioner none;minIter 1000; maxIter 10000;".format(
            matrix_solver)
        sed(fn, "p{}", solver_str)

    def run(self, results_accumulator):
        accumulated_time = 0
        max_time = 2 * 60
        while (accumulated_time < max_time):
            start = datetime.datetime.now()
            ret = check_output([self.of_solver], cwd=self.path / "boxTurb16")
            end = datetime.datetime.now()
            run_time = (end - start).total_seconds()
            self.results_accumulator.add(run_time)
            accumulated_time += run_time
        print(ret)


def setup_base_case(test_path, cells):
    """ """
    import os
    test_path = test_path / str(cells)
    ensure_path(test_path)
    foam_tutorials = Path(os.environ["FOAM_TUTORIALS"])
    case_name = "boxTurb16"
    foam_base_location = foam_tutorials / "DNS" / "dnsFoam" / case_name
    base_case = test_path / case_name
    base_case_system_folder = base_case / "system"
    print("copying {} to {}".format(foam_base_location, test_path))
    check_output(["cp", "-r", foam_base_location, test_path])

    new_cells = "{} {} {}".format(cells, cells, cells)
    blockMeshDict = base_case_system_folder / "blockMeshDict"
    set_cells(blockMeshDict, "16 16 16", new_cells)

    controlDict = base_case_system_folder / "controlDict"
    add_libOGL_so(controlDict)
    set_end_time(controlDict, 1.0)
    set_deltaT(controlDict, 1.0)

    fvSolution = base_case_system_folder / "fvSolution"
    clear_solver_settings(fvSolution)

    print("Meshing")
    check_output(["blockMesh"], cwd=base_case)


if __name__ == "__main__":

    arguments = docopt(__doc__, version='runBench 0.1')

    # TODO replace by class
    executor = [("base", "", "")]

    solver = ["CG"]

    preconditioner = []  # "none", "Jacobi"]

    if arguments["--cuda"]:
        executor.append(("GKO", "GKO", "CUDA"))

    if arguments["--of"]:
        executor.append(("OF", "P", ""))

    if arguments["--ref"]:
        executor.append(("GKO", "GKO", "ref"))

    if arguments["--omp"]:
        executor.append(("GKO", "GKO", "OMP"))

    number_of_cells = [8, 16, 32]  # , 64, 128]

    if arguments["--large-cases"]:
        number_of_cells += [64, 128]

    test_path = Path(arguments["--folder"])

    results = Results(arguments["--report"])

    for e in executor:
        for s in solver:
            for n in number_of_cells:
                print(e, s, n)
                # check if path exist and clean is set
                path = test_path / Path(e[0] + e[1]) / str(n)
                exist = os.path.isdir(path)
                skip = False
                clean = arguments["--clean"]
                if exist and clean:
                    shutil.rmtree(path)
                    skip = False
                if exist and not clean:
                    skip = True
                if e[0] == "base":
                    if not skip:
                        setup_base_case(test_path / Path(e[0] + e[1]), n)
                    print("skipping")
                else:
                    if not skip:
                        case = Case(
                            test_base=test_path,
                            of_solver="dnsFoam",
                            solver=s,
                            executor=e,
                            base_case=test_path / "base" / str(n) /
                            "boxTurb16",
                            resolution=n,
                            results=results,
                        )
                        case.run(results)
                    print("skipping")
