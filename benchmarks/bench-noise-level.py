from rs import *
import graphix
import sys
import os

if __name__ == "__main__":
    level = float(sys.argv[1])
    nSimulations = 15
    ts = TimeSuite(nSimulations=nSimulations, noise_level=level)

    ts.test_consistency()

    result_np = benchmark(ts, graphix.sim.density_matrix.DensityMatrix, "density_matrix")
    result_rs = benchmark(ts, graphix.sim.density_matrix.RustDensityMatrix, "density_matrix")

    total_time_np = result_np.total_tt / nSimulations
    total_time_rs = result_rs.total_tt / nSimulations

    output_dir = "bench_noise_outputs"
    os.makedirs(output_dir, exist_ok=True)

    out_path = f"{output_dir}/{str(level)}"
    with open(out_path, "w") as out:    # Write the result to a file
        out.write(f"{total_time_np},{total_time_rs}")