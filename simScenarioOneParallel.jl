# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/30/2015

parallel = :both
ncpu_local = CPU_CORES / 2
machines = [("youngjun@cambridge", 6, "/usr/bin"), ("youngjun@cheonan", 4, "/usr/bin")]


include("ScenarioOneIJUtil.jl")

if parallel == :local_ || parallel == :both
    addprocs(int64(ncpu_local))
end

if parallel == :remote || parallel == :both
    for (machine, count, dir) in machines
        cluster_list = ASCIIString[]

        for i = 1:count
            push!(cluster_list, machine)
        end

        addprocs(cluster_list, dir = dir)
    end
end

@everywhere __PARALLEL__ = true
require("ScenarioOne_.jl")
require("simScenarioOne.jl")


function buildDatabaseV0_1(datafile::ASCIIString; update::Bool = false, bParallel::Bool = false)

    for param_set in [1, 2]
        for policy in [:stay, :back, :landing]
            for aircraft_traj_uncertainty in [0., 1., 2.]
                retrieveEvaluation(param_set, policy, datafile = datafile, update = update, aircraft_traj_uncertainty = aircraft_traj_uncertainty, N_min = 1000, N_max = 10000, RE_threshold = 0.01, bParallel = bParallel)
            end
        end
    end
end


function buildDatabaseV0_2(datafile::ASCIIString; update::Bool = false, bParallel::Bool = false)

    for param_set in [1, 2]
        for policy in [:stay, :back, :landing, :lower]
            for sim_time_mu in [5., 10., 15.]
                retrieveEvaluation(param_set, policy, datafile = datafile, update = update, sim_time_mu = sim_time_mu, N_min = 1000, N_max = 10000, RE_threshold = 0.01, bParallel = bParallel)
            end
        end
    end
end


buildDatabaseV0_1("s1results_v0_1.jld" * "." * string(int64(time())), update = false, bParallel = true)
buildDatabaseV0_2("s1results_v0_2.jld" * "." * string(int64(time())), update = false, bParallel = true)


