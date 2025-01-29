import random as rd

from graphix.noise_models import DepolarisingNoiseModel
from graphix.sim.density_matrix import DensityMatrixBackend, DensityMatrix, RustDensityMatrix
from graphix import Circuit
import graphix.command
from graphix.states import BasicStates

from graphix.random_objects import rand_circuit

from veriphix.client import Client, Secrets

import cProfile, pstats, io

def create_pattern():
    circuit = Circuit(2)
    circuit.cnot(0, 1)
    circuit.h(0)
    circuit.h(1)
    pattern = circuit.transpile().pattern  ## 6 nodes
    pattern.minimize_space()

    ## Measure output nodes, to have classical output
    classical_output = pattern.output_nodes
    for onode in classical_output:
        pattern.add(graphix.command.M(node=onode))

    pattern.standardize()
    return classical_output, pattern

def vbqc_simulation(impl=DensityMatrix, noise_level=0.):
    secrets = Secrets(r=True, a=True, theta=True)

    classical_output, pattern = create_pattern()
    states = [BasicStates.ZERO for _ in range(2)]
    client = Client(pattern=pattern, secrets=secrets, input_state=states)

    test_runs = client.create_test_runs()
    # Trappified scheme parameters
    d = 50  # nr of computation rounds
    t = 50  # nr of test rounds
    N = d + t
    rounds = list(range(N))
    rd.shuffle(rounds)
    test_runs = client.create_test_runs()

    # Store data for each value of p
    all_histograms = {}
    failed_traps_histograms = {}

    # Defining the noise model (depolarizing noise)
    noise = DepolarisingNoiseModel(entanglement_error_prob=noise_level)

    # Recording outcomes/traps failures
    outcomes_histogram = dict()
    n_failed_traps = 0

    backend = DensityMatrixBackend(impl=impl)
    print(f"============================ VBQC simulation with {backend.state} ============================")

    # Iterating through rounds
    for i in rounds:
        if i < d:
            # Computation round
            client.refresh_randomness()
            client.delegate_pattern(backend=backend, noise_model=noise)

            # Store result (increment occurrence in histogram)
            computation_outcome = ""
            for onode in classical_output:
                computation_outcome += str(int(client.results[onode]))
            if computation_outcome not in outcomes_histogram:
                outcomes_histogram[computation_outcome] = 1
            else:
                outcomes_histogram[computation_outcome] += 1
        else:
            # Test round
            run = rd.choice(test_runs)
            client.refresh_randomness()
            trap_outcomes = client.delegate_test_run(run=run, backend=backend, noise_model=noise)

            # Record trap failure
            # A trap round fails if one of the single-qubit traps failed
            if sum(trap_outcomes) != 0:
                n_failed_traps += 1

    # Combine results
    all_histograms[noise_level] = outcomes_histogram
    if t != 0:
        failed_traps_histograms[noise_level] = n_failed_traps / (t)
    
    all_outcomes = sorted(set().union(*[hist.keys() for hist in all_histograms.values()]))

    return all_outcomes



class TimeSuite:
    def __init__(self, nSimulations=10, noise_level=0.):
        self.nSimulations = nSimulations
        self.noise_level = noise_level


    def test_consistency(self):
        for _ in range(self.nSimulations):
            np_result = vbqc_simulation(impl=DensityMatrix, noise_level=self.noise_level)
            rust_result = vbqc_simulation(impl=RustDensityMatrix, noise_level=self.noise_level)
            assert np_result == rust_result

    def time_impl(self, impl):
        for _ in range(self.nSimulations):
            vbqc_simulation(impl=impl, noise_level=self.noise_level)

ts = TimeSuite(nSimulations=5, noise_level=0.15)
ts.test_consistency()

def benchmark(ts, impl, identifier=""):
    pr = cProfile.Profile()
    pr.enable()
    ts.time_impl(impl)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(identifier)
    print(s.getvalue())


benchmark(ts, graphix.sim.density_matrix.DensityMatrix, "trappifiedCanvas|client|density_matrix")
benchmark(ts, graphix.sim.density_matrix.RustDensityMatrix, "trappifiedCanva|client|density_matrix")