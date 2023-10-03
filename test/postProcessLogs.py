import os
import sys
import json

from pathlib import Path
from subprocess import check_output
from obr.OpenFOAM.case import OpenFOAMCase
from obr.core.core import find_solver_logs
from copy import deepcopy
from Owls.parser.LogFile import LogFile

def call(jobs):
    """Based on the passed jobs all existing log files are parsed the
    results get stored in the job document using the following schema

     obr:
         postprocessing:
           cells:
           decomposition: {}
           ...
           runs: [
             - {host: hostname, timestamp: timestamp, results: ...} <- records
            ]
    """
    for job in jobs:
        job.doc["data"] = []

        run_logs = []
        # find all solver logs corresponding to this specific jobs
        for log, campaign, tags in find_solver_logs(job):

            cont_error = [                                                                                                     
                LogKey("time step continuity errors",                                                                   ["ContinuityError " + i for i in ["local", "global", "cumulative"]],                                                    
                )                     
            ]

            log_file_parser = LogFile(log_keys)
            df = log_file_parser.parse_to_df(log)

            run_logs.append({"ContinuityError": df["ContinuityError_cum"].values[-1]})

        # store all records
        job.doc["data"] = run_logs
