# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/30/2015

parallel = :local_
ncpu_local = CPU_CORES / 2
machines = [("youngjun@tula", 20, "/usr/bin"), ("youngjun@cheonan", 4, "/usr/bin"), ("youngjun@cambridge", 6, "/usr/bin")]


__PARALLEL__ = true
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


function buildDatabaseV0_1(datafile::ASCIIString; update::Bool = false, MS::Bool = false, bParallel::Bool = false)

    println(strftime(time()), " buildDatabaseV0_1")

    for param_set in [1, 2]
        for policy in [:stay, :back, :landing]
            for aircraft_traj_uncertainty in [0., 1., 2.]
                println(strftime(time()), " ", param_set, " ", policy, " ", aircraft_traj_uncertainty)
                retrieveEvaluation("0.1", param_set, policy, datafile = datafile, update = update, aircraft_traj_uncertainty = aircraft_traj_uncertainty, N_min = 1000, N_max = 10000, RE_threshold = 0.01, MS = MS, bParallel = bParallel)
            end
        end
    end
end


function buildDatabaseV0_2(datafile::ASCIIString; update::Bool = false, MS::Bool = false, bParallel::Bool = false)

    println(strftime(time()), " buildDatabaseV0_2")

    for param_set in [1, 2]
        for policy in [:stay, :back, :landing, :lower]
            for sim_comm_loss_duration_mu in [5., 10., 15.]
                for sim_comm_loss_duration_sigma in [0., 1., 2., 5.]
                    println(strftime(time()), " ", param_set, " ", policy, " ", sim_comm_loss_duration_mu, " ", sim_comm_loss_duration_sigma)
                    retrieveEvaluation("0.2", param_set, policy, datafile = datafile, update = update, sim_comm_loss_duration_mu = sim_comm_loss_duration_mu, sim_comm_loss_duration_sigma = sim_comm_loss_duration_sigma, N_min = 1000, N_max = 10000, RE_threshold = 0.01, MS = MS, bParallel = bParallel)
                end
            end
        end
    end
end


function buildDatabaseV1_0(datafile::ASCIIString; update::Bool = false, MS::Bool = false, bParallel::Bool = false)

    println(strftime(time()), " buildDatabaseV1_0")

    for param_set in [1, 2]
        for policy in [:stay, :back, :landing, :lower]
            for r_surveillance in [10., 20., 30.]
                for uav_surveillance_pattern in [:mower, :chase, :back]
                    println(strftime(time()), " ", param_set, " ", policy, " ", r_surveillance, " ", uav_surveillance_pattern)
                    retrieveEvaluation("1.0", param_set, policy, datafile = datafile, update = update, r_surveillance = r_surveillance, uav_surveillance_pattern = uav_surveillance_pattern, N_min = 1000, N_max = 10000, RE_threshold = 0.01, MS = MS, bParallel = bParallel)
                end
            end
        end
    end
end


#buildDatabaseV0_1("s1results_v0_1.jld" * "." * string(int64(time())), update = false, bParallel = true)

#buildDatabaseV0_2("s1results_v0_2.jld" * "." * string(int64(time())), update = false, MS = true, bParallel = true)

#buildDatabaseV1_0("s1results_v1_0.jld" * "." * string(int64(time())), update = false, MS = true, bParallel = true)


