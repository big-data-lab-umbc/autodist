import itertools
import subprocess

import os

from .single_run import STRATEGIES_FOR_DISTRIBUTED_TESTS

cases = [
    "c0",  # TensorFlow 2.0 basics
    "c1",  # Keras basics
    "c2",  # Sparse basics
    "c3"  # Numpy basics
]

resource_specs = [
    os.path.join(os.path.dirname(__file__), 'resource_specs/r1.yml'),
    os.path.join(os.path.dirname(__file__), 'resource_specs/r3.yml')
]


def test_dist():
    combinations = itertools.product(resource_specs, STRATEGIES_FOR_DISTRIBUTED_TESTS.keys(), cases)
    for r, s, c in combinations:
        if s == 'AllReduce' and c not in ["c0", "c1"]:
            continue
        cmd = ("python /home/autodist/autodist/tests/integration/single_run.py "
               "--case={} --strategy={} --resource={}").format(c, s, r)
        print("=====> test starts!")
        print("=====> cmd is {}".format(cmd))
        status = subprocess.run(args=cmd, shell=True)
        assert status.returncode == 0, "{}, {}, {}".format(r, s, c)
        print("=====> test success")
