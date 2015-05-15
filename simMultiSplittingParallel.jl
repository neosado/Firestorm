# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 04/30/2015

parallel = :remote
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
require("simMultiSplitting.jl")


simMultiSplitting()


