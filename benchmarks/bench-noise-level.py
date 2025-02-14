from veriphix.benchmarks.rs import *
import graphix
import sys

if __name__ == "__main__":
    level = float(sys.argv[1])
    ts = TimeSuite(nSimulations=10, noise_level=level)
    ts.test_consistency()
    
    benchmark(ts, graphix.sim.density_matrix.DensityMatrix, "density_matrix")
    benchmark(ts, graphix.sim.density_matrix.RustDensityMatrix, "density_matrix")