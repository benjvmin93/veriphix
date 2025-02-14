import graphix.command
from graphix.random_objects import Circuit, rand_circuit
from graphix.states import BasicStates
from graphix.sim.density_matrix import DensityMatrixBackend, DensityMatrix, RustDensityMatrix
from graphix.sim.statevec import StatevectorBackend, Statevec
from graphix.pauli import Pauli
from graphix.fundamentals import IXYZ

import stim

from graphix.noise_models import DepolarisingNoiseModel
from veriphix.client import Secrets, Client

import numpy as np

import pstats
import io
import cProfile
import sys

def run_delegate(client, backend, noise):
    return client.delegate_pattern(backend=backend, noise_model=noise)

class TimeSuite:
    def __init__(self, nqubits=7, nSimulations=10, depth=1, noise=None):
        print(f"Running benchmark with {nSimulations} circuits of {nqubits} qubits, depth {depth} and {noise} noise model")
        self.noise = noise
        circuits = [rand_circuit(nqubits, depth) for _ in range(nSimulations)]
        self.pat = [circ.transpile().pattern for circ in circuits]
        for pat in self.pat:
            print(f"pat.max_space() before : {pat.max_space()}")
            pat.minimize_space()
            print(f"pat.max_space() after : {pat.max_space()}")
            for onode in pat.output_nodes:
                pat.add(graphix.command.M(node=onode))

    def test_consistency(self):
        secrets = Secrets(r=True, a=True, theta=True)
        print(f"Asserting consistency between the two implementations.")
        for pat in self.pat:
            print(pat)
            client = Client(pat, secrets=secrets)
            np_outcome = run_delegate(client, DensityMatrixBackend(impl=DensityMatrix), self.noise)
            rs_outcome = run_delegate(client, DensityMatrixBackend(impl=RustDensityMatrix), self.noise)
            assert np_outcome == rs_outcome
            print(".")

    def time_impl(self, impl):
        secrets = Secrets(r=True, a=True, theta=True)
        for pat in self.pat:
            print(".")
            client = Client(pat, secrets=secrets)
            run_delegate(client, DensityMatrixBackend(impl=impl), self.noise)
            
def benchmark(ts, impl, identifier=""):
    pr = cProfile.Profile()
    pr.enable()
    ts.time_impl(impl)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(identifier)
    print(s.getvalue())


if __name__ == "__main__":
    depth = int(sys.argv[1])
    
    noise = DepolarisingNoiseModel(entanglement_error_prob=0.01)
    ts = TimeSuite(depth=depth, noise=noise)
    ts.test_consistency()

    benchmark(ts, DensityMatrix, "density_matrix")
    benchmark(ts, RustDensityMatrix, "density_matrix")