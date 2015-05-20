# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 04/30/2015

require("simScenarioOne.jl")


function simMultiSplitting()

    srand(uint(time()))

    version = "1.0"
    param_set = 1

    params = generateParams(param_set)

    if version == "0.1"
        params.sim_comm_loss_duration_mu = 10.

        params.wf_init_loc = [(6, 4), (7, 4), (5, 5), (6, 5), (7, 5), (4, 6), (5, 6), (6, 6), (7, 6), (5, 7), (6, 7), (6, 8)]
        params.wf_sim_time = 0
        params.wf_p_fire = 0.06

        params.uav_loc = (4, 5)
        # :stay, :back, :landing
        params.uav_policy = :back

        params.aircraft_traj_uncertainty = 1.

    elseif version == "0.2"
        # 68% within 1 standard deviation
        # 95% within 2 standard deviation
        # 99.7% within 3 standard deviation
        params.sim_comm_loss_duration_mu = 10.
        params.sim_comm_loss_duration_sigma = 1.

        params.wf_init_loc = [(6, 4), (7, 4), (5, 5), (6, 5), (7, 5), (4, 6), (5, 6), (6, 6), (7, 6), (5, 7), (6, 7), (6, 8)]
        params.wf_sim_time = 0
        params.wf_p_fire = 0.06

        params.uav_loc = (4, 5)
        # :stay, :back, :landing, :lower
        params.uav_policy = :back

        params.aircraft_traj_uncertainty = 0.
        params.aircraft_traj_adaptive = true
        params.aircraft_operation_time_limit = 30

    elseif version == "1.0"
        params.sim_time = 30
        params.sim_comm_loss_duration_mu = 10.
        params.sim_comm_loss_duration_sigma = 1.
        params.sim_continue = true

        params.wf_init_loc = [(6, 4), (7, 4), (5, 5), (6, 5), (7, 5), (4, 6), (5, 6), (6, 6), (7, 6), (5, 7), (6, 7), (6, 8)]
        params.wf_sim_time = 0
        params.wf_p_fire = 0.06

        params.r_surveillance = 10.

        params.uav_loc = (4, 5)
        # :stay, :back, :landing, :lower
        params.uav_policy = :back
        # :mower, :chase, :back
        params.uav_surveillance_pattern = :mower
        # :lower
        params.uav_cas = :lower

        params.aircraft_traj_uncertainty = 0.
        params.aircraft_traj_adaptive = true
        params.aircraft_operation_time_limit = 0

    end

    if false
        params.uav_loc = (9, 4)
        params.uav_policy = :stay
        #params.sim_comm_loss_duration_sigma = 0.
        #params.r_dist = [1. 0.; 2. -100.; 3. -20.]

        #result = estimateExpectedUtility(params, N_min = 1000, N_max = 1000000, RE_threshold = 0.01, MS = true, verbose = 1)
        #println(result)


        #L = nothing
        #R = 1000000

        #L = [Inf, 3] # level 0, level 1, ...
        #R = [700000, 10] # level 0, level 1, ...

        #L = [Inf, 3, 2] # level 0, level 1, ...
        #R = [100000, 7, 20] # level 0, level 1, ...

        #estimateExpectedUtilityMS(params, L, R, RE_threshold = 0.01, verbose = 1)


        #lst = [(nothing, 1000), ([Inf, 3], [1000, 10])]

        #lst = [(nothing, 1000000), (nothing, 1000000), (nothing, 1000000), (nothing, 1000000), (nothing, 1000000),
        #       ([Inf, 2], [1000000, 10]), ([Inf, 2], [1000000, 20]), ([Inf, 2], [1000000, 30]), ([Inf, 2], [1000000, 40]), ([Inf, 2], [1000000, 50]),
        #       ([Inf, 3], [1000000, 10]), ([Inf, 3], [1000000, 20]), ([Inf, 3], [1000000, 30]), ([Inf, 3], [1000000, 40]), ([Inf, 3], [1000000, 50]),
        #       ([Inf, 4], [1000000, 10]), ([Inf, 4], [1000000, 20]), ([Inf, 4], [1000000, 30]), ([Inf, 4], [1000000, 40]), ([Inf, 4], [1000000, 50]),
        #       ([Inf, 5], [1000000, 10]), ([Inf, 5], [1000000, 20]), ([Inf, 5], [1000000, 30]), ([Inf, 5], [1000000, 40]), ([Inf, 5], [1000000, 50])]

        lst = {}
        for i = 1:30
            push!(lst, (nothing, 1000000))
        end
        for i = 1:30
            push!(lst, ([Inf, 3], [1000000, 10]))
        end

        #results = {}
        #for x in lst
        #    LOG = estimateExpectedUtilityMS(params, x[1], x[2], RE_threshold = 0.01, verbose = 1, bLog = true)
        #    push!(results, LOG)
        #end

        results = pmap(x -> estimateExpectedUtilityMS(params, x[1], x[2], RE_threshold = 0.01, bLog = true), lst)

        list = readdir("mslogs")
        if length(list) == 0
            k = 1
        else
            k = sort(map(x -> int(split(x, ".")[1][3:end]), list))[end] + 1
        end

        i = 1
        for (L, R) in lst
            f = open("mslogs/ms$k.json", "w")
            JSON.print(f, {(L, R), results[i]})
            close(f)

            i += 1
            k += 1
        end
    end

    if false
        params.uav_loc = (9, 4)
        params.uav_policy = :stay
        #params.sim_comm_loss_duration_sigma = 0.

        lst = [1:30]
        results = pmap(x -> estimateExpectedUtility(params, N_min = 1000, N_max = 100000, RE_threshold = 0.01, MS = true, bLog = true), lst)

        list = readdir("mslogs")
        if length(list) == 0
            k = 1
        else
            k = sort(map(x -> int(split(x, ".")[1][3:end]), list))[end] + 1
        end

        for i in lst
            f = open("mslogs/ms$k.json", "w")
            JSON.print(f, results[i])
            close(f)

            k += 1
        end
    end

end


#simMultiSplitting()


