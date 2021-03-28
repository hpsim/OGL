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
        --of                Generate default of cases [default: False].
        --ref               Generate ref cases [default: False].
        --cuda              Generate cuda cases [default: False].
        --omp               Generate omp cases [default: False].
        --clean             Remove existing cases [default: False].
        --large-cases       Include large cases [default: False].

"""
from docopt import docopt
from subprocess import check_output
from pathlib import Path
import os
import shutil
import datetime
from itertools import product


class Results:
    def __init__(self, fn):
        self.fn = Path(fn)
        self.columns = [
            "domain",
            "executor",
            "solver",
            "number_of_iterations",
            "resolution",
            "run_time",
        ]
        self.current_col_vals = []
        self.report_handle = open(self.fn, "a+")

    def set_case(self, domain, executor, solver, number_of_iterations, resolution):
        self.current_col_vals = [
            domain,
            executor,
            solver,
            number_of_iterations,
            resolution,
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
    ret = check_output(["sed", "-i", "s/" + in_reg_exp + "/" + out_reg_exp + "/g", fn])


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
    # sed(fvSolution, "p\\n[ ]*[{][^}]*[}]", "p{}")
    clean_block_from_file(fvSolution, "   p\n", "  }\n", "p{}\n")


def ensure_path(path):
    print("creating", path)
    check_output(["mkdir", "-p", path])


class Case:
    def __init__(
        self,
        test_base=None,
        solver="CG",
        executor=None,
        base_case=None,
        resolution=32,
        results=None,
        iterations=0,
        is_base_case=False,
        of_tutorial_domain="DNS",
        of_solver="dnsFoam",
        of_tutorial_case="boxTurb16",
    ):
        self.variable = None
        self.is_base_case = is_base_case
        self.test_base = test_base
        self.of_base_case = "boxTurb16"
        self.fields = "p"
        self.resolution = resolution
        self.executor = executor
        self.solver = solver
        self.iterations = iterations
        self.base_case_path_ = base_case
        self.results_accumulator = results
        self.init_time = 0
        self.of_solver = of_solver
        self.of_tutorial_case = of_tutorial_case
        self.of_tutorial_domain = of_tutorial_domain

    @property
    def system_folder(self):
        return self.path / "system"

    @property
    def controlDict(self):
        return self.system_folder / "controlDict"

    @property
    def blockMeshDict(self):
        return self.system_folder / "blockMeshDict"

    @property
    def fvSolution(self):
        return self.system_folder / "fvSolution"

    def create(self):
        ensure_path(self.parent_path)
        self.copy_base(self.base_case_path, self.parent_path)
        deltaT = 0.1 * 16 / self.resolution
        if self.is_base_case:
            new_cells = "{} {} {}".format(
                self.resolution, self.resolution, self.resolution
            )
            set_cells(self.blockMeshDict, "16 16 16", new_cells)
            add_libOGL_so(self.controlDict)
            set_end_time(self.controlDict, 10 * deltaT)
            set_deltaT(self.controlDict, deltaT)
            clear_solver_settings(self.fvSolution)
            print("Meshing", self.path)
            check_output(["blockMesh"], cwd=self.path)
            return

        self.set_matrix_solver(self.fvSolution)

    @property
    def base_case_path(self):
        if self.is_base_case:
            foam_tutorials = Path(os.environ["FOAM_TUTORIALS"])
            # return (
            #     foam_tutorials
            #     / self.of_tutorial_domain
            #     / self.of_solver
            #     / self.of_tutorial_case
            # )
            return Path("Test") / self.of_tutorial_case
        return self.base_case_path_ / self.of_base_case

    @property
    def parent_path(self):
        return (
            Path(self.test_base)
            / Path(self.executor.local_path)
            / self.fields
            / str(self.resolution)
        )

    @property
    def path(self):
        return self.parent_path / str(self.of_tutorial_case)

    def copy_base(self, src, dst):
        print("copying base case", src, dst)
        check_output(["cp", "-r", src, dst])

    def set_matrix_solver(self, fn):
        print("setting solver", fn)
        matrix_solver = self.executor.prefix + self.solver
        solver_str = (
            "p{"
            + "solver {};tolerance 1e-06; relTol 0.0;smoother none;preconditioner none;minIter {}; maxIter 10000;".format(
                matrix_solver, self.iterations
            )
        )
        sed(fn, "p{}", solver_str)

    def run(self, results_accumulator):
        if self.is_base_case:
            return
        self.results_accumulator.set_case(
            domain=self.executor.domain,
            executor=self.executor.executor,
            solver=self.solver,
            number_of_iterations=self.iterations,
            resolution=self.resolution,
        )
        accumulated_time = 0
        max_time = 100.0
        iters = 0
        while accumulated_time < max_time or iters < 3:
            iters += 1
            start = datetime.datetime.now()
            ret = check_output([self.of_solver], cwd=self.path)
            end = datetime.datetime.now()
            run_time = (end - start).total_seconds() - self.init_time
            self.results_accumulator.add(run_time)
            accumulated_time += run_time


def build_parameter_study(test_path, results, executor, solver, setter):
    for (e, s, n) in product(executor, solver, setter):
        # check if path exist and clean is set
        path = test_path / e.local_path / str(n.value)
        exist = os.path.isdir(path)
        skip = False
        clean = arguments["--clean"]
        if exist and clean:
            shutil.rmtree(path)
            skip = False
        if exist and not clean:
            skip = True
        is_base_case = False
        base_case_path = test_path / Path("base") / Path("p") / str(n.value)

        if e.domain == "base":
            print("is base case")
            is_base_case = True

        if not skip:
            case = Case(
                test_base=test_path,
                solver=s,
                executor=e,
                base_case=base_case_path,
                results=results,
                is_base_case=is_base_case,
            )
            n.run(case)
            case.create()
            case.run(results)
        else:
            print("skipping")


def resolution_study(name, executor, solver, arguments):

    test_path = Path(arguments["--folder"]) / name

    results = Results(arguments["--report"])

    number_of_cells = [8, 16, 32]

    if arguments["--large-cases"]:
        number_of_cells += [64, 128]

    n_setters = []
    for n in number_of_cells:
        n_setters.append(ValueSetter("resolution", n))

    build_parameter_study(test_path, results, executor, solver, n_setters)


class ValueSetter:
    def __init__(self, prop, value):
        self.prop = prop
        self.value = value

    def run(self, case):
        setattr(case, self.prop, self.value)
        setattr(case, "variable", self.value)


class Executor:
    def __init__(self, domain, solver_prefix, executor=None, cmd_prefix=None):
        self.domain = domain
        self.prefix = solver_prefix
        self.executor = executor
        self.cmd_prefix = cmd_prefix

    @property
    def local_path(self):
        path = self.domain
        if self.executor:
            path += self.executor
        return Path(path)


if __name__ == "__main__":

    arguments = docopt(__doc__, version="runBench 0.1")

    executor = [Executor("base", "", "")]

    solver = ["CG", "BiCGStab"]

    preconditioner = []

    if arguments["--cuda"]:
        executor.append(Executor("GKO", "GKO", "CUDA"))

    if arguments["--of"]:
        executor.append(Executor("OF", "P", ""))

    if arguments["--ref"]:
        executor.append(Executor("GKO", "GKO", "ref"))

    if arguments["--omp"]:
        executor.append(Executor("GKO", "GKO", "OMP"))

    resolution_study("number_of_cells", executor, solver, arguments)
