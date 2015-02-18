# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/13/2015

global __PARALLEL__
if !isdefined(:__PARALLEL__) 
    __PARALLEL__ = false
end

using ScenarioOne_
if !__PARALLEL__
using ScenarioOneVisualizer_
end

using Base.Test


function generateParams(setnum::Int64 = 1)

    params = ScenarioOneParams()

    params.n = 11

    params.sim_time = params.n * params.n
    params.sim_continue = false

    params.sim_comm_loss_duration_mu = 0.
    params.sim_comm_loss_duration_sigma = 0.

    params.wf_init_loc = (int(params.n / 2), int(params.n / 2))
    params.wf_sim_time = params.n
    params.wf_p_fire = 0.06

    params.p_uav_landing_crash_rate = 0.01
    params.p_uav_lower_crash_rate = 0.02

    params.r_crashed = -15.
    params.r_dist = [1. -100.; 2. -20.; 3. -10.]
    params.r_surveillance = 1.

    params.uav_loc = (4, 5)
    params.uav_base_loc = (11, 8)
    params.uav_velocity = 0.3
    params.uav_policy = :back
    params.uav_cas = nothing

    if setnum == 1
        params.aircraft_start_loc = (3, 11)
        params.aircraft_end_loc = (10, 1)
        params.aircraft_control_points = [(6.5, 6.)]
    elseif setnum == 2
        params.aircraft_start_loc = (4, 11)
        params.aircraft_end_loc = (7, 11)
        params.aircraft_control_points = [(-2., 1.), (13., 1.)]
    else
        error("invalid parameter set")
    end

    params.aircraft_velocity = 1.
    params.aircraft_traj_uncertainty = 0.

    params.aircraft_traj_adaptive = false
    params.aircraft_operation_time_limit = 0

    return params
end


function simulate(params::ScenarioOneParams; draw::Bool = false, wait::Bool = false)

    s1 = ScenarioOne(params)
    state = ScenarioOneState(s1)

    initTrajectories(s1)

    if draw
        s1v = ScenarioOneVisualizer(wait = wait)

        visInit(s1v, s1)
        visUpdate(s1v, s1)
        updateAnimation(s1v)
    end

    t = 0
    U = 0.

    while !isEndState(s1, state, t)
        updateStatePartA(s1, state, t)

        r = getReward(s1, state, t)
        U += r

        if draw
            visInit(s1v, s1, t)
            visUpdate(s1v, s1, t, state, r, U)
            updateAnimation(s1v, t)
        end

        updateStatePartB(s1, state, t)

        t += 1
    end

    if draw
        saveAnimation(s1v, repeat = true)
    end

    return U
end


function estimateExpectedUtility(params::ScenarioOneParams; N_min::Int = 0, N_max::Int = 1000, RE_threshold::Float64 = 0., verbose::Int64 = 0)

    meanU = 0.
    ssU = 0.
    RE = 0.

    n = 0
    for i = 1:N_max
        n += 1

        U = simulate(params)

        meanU += (U - meanU) / i
        ssU += U * U
        if i > 1
            var = (ssU - i * (meanU * meanU)) / ((i - 1) * i)
            RE = abs(sqrt(var) / meanU)

            if verbose >= 1 && i % 100 == 0
                println("i: $i, mean: $meanU, RE: $RE")
            end

            if i >= N_min && (isnan(RE) || (RE_threshold != 0 && RE < RE_threshold))
                break
            end
        end
    end

    if verbose >= 1
        println("n: $n, mean: $meanU, RE: $RE")
    end

    return meanU, RE, n
end


function evaluatePolicy(version::ASCIIString, param_set_num::Int64, policy::Symbol; sim_comm_loss_duration_mu::Union(Float64, Nothing) = nothing, sim_comm_loss_duration_sigma::Float64 = 0., sim_continue::Bool = false, r_surveillance::Float64 = 0., uav_surveillance_pattern::Union(Symbol, Nothing) = nothing, aircraft_traj_uncertainty::Union(Float64, Nothing) = nothing, N_min::Int = 0, N_max::Int = 1000, RE_threshold::Float64 = 0., bParallel::Bool = false)

    params = generateParams(param_set_num)

    params.uav_policy = policy

    if version == "0.1"
        @assert aircraft_traj_uncertainty != nothing

        params.wf_sim_time = params.n
        params.wf_p_fire = 0.12

        params.aircraft_traj_uncertainty = aircraft_traj_uncertainty

    elseif version == "0.2" || version == "0.2.1"
        @assert sim_comm_loss_duration_mu != nothing

        params.sim_comm_loss_duration_mu = sim_comm_loss_duration_mu
        params.sim_comm_loss_duration_sigma = sim_comm_loss_duration_sigma

        params.wf_sim_time = int(params.n * 2)
        params.wf_p_fire = 0.06

        params.aircraft_traj_adaptive = true
        params.aircraft_operation_time_limit = 30

    elseif version == "0.3"
        @assert sim_continue

        params.sim_time = 30
        params.sim_comm_loss_duration_mu = 10.
        params.sim_comm_loss_duration_sigma = 1.
        params.sim_continue = true

        params.wf_sim_time = params.n * 2
        params.wf_p_fire = 0.06

        params.r_surveillance = r_surveillance

        params.aircraft_traj_adaptive = true
        params.aircraft_operation_time_limit = 0

    elseif version == "1.0"
        @assert uav_surveillance_pattern != nothing

        params.sim_time = 30
        params.sim_comm_loss_duration_mu = 10.
        params.sim_comm_loss_duration_sigma = 1.
        params.sim_continue = true

        params.wf_sim_time = params.n * 2
        params.wf_p_fire = 0.06

        params.r_surveillance = 0.5

        params.uav_surveillance_pattern = uav_surveillance_pattern
        params.uav_cas = :lower

        params.aircraft_traj_adaptive = true
        params.aircraft_operation_time_limit = 0

    end


    U = zeros(params.n, params.n)
    RE = zeros(params.n, params.n)
    N = zeros(Int64, params.n, params.n)

    if bParallel == false
        for i = 1:params.n
            for j = 1:params.n
                params.uav_loc = (i, j)

                U[i, j], RE[i, j], N[i, j] = estimateExpectedUtility(params, N_min = N_min, N_max = N_max, RE_threshold = RE_threshold)

                #println("($i, $j), mean: ", int(U[i, j]), ", RE: ", signif(RE[i, j], 4), ", N: ", N[i, j])
            end
        end

    else
        lst = (Int64, Int64)[]
        for j = 1:params.n
            for i = 1:params.n
                push!(lst, (i, j))
            end
        end

        results = pmap(x -> (params.uav_loc = x; estimateExpectedUtility(params, N_min = N_min, N_max = N_max, RE_threshold = RE_threshold)), lst)

        k = 1
        for result in results
            U[k] = result[1]
            RE[k] = result[2]
            N[k] = result[3]

            k += 1
        end
    end

    # debug
    #println(params)
    #println(int(U))
    #println(round(RE, 4))
    #println(N)

    return U, RE, N, params
end


if false
    srand(uint(time()))

    version = "1.0"
    param_set = 1

    params = generateParams(param_set)

    if version == "0.1"
        params.sim_comm_loss_duration_mu = params.n

        params.wf_p_fire = 0.12

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

        params.wf_sim_time = params.n * 2
        params.wf_p_fire = 0.06

        params.uav_loc = (4, 5)
        # :stay, :back, :landing, :lower
        params.uav_policy = :back

        params.aircraft_traj_uncertainty = 0.
        params.aircraft_traj_adaptive = true
        params.aircraft_operation_time_limit = 30

    elseif version == "0.3"
        params.sim_time = 30
        params.sim_comm_loss_duration_mu = 10.
        params.sim_comm_loss_duration_sigma = 1.
        params.sim_continue = true

        params.wf_sim_time = params.n * 2
        params.wf_p_fire = 0.06

        params.r_surveillance = 1.

        params.uav_loc = (4, 5)
        # :stay, :back, :landing, :lower
        params.uav_policy = :back

        params.aircraft_traj_uncertainty = 0.
        params.aircraft_traj_adaptive = true
        params.aircraft_operation_time_limit = 0

    elseif version == "1.0"
        params.sim_time = 30
        params.sim_comm_loss_duration_mu = 10.
        params.sim_comm_loss_duration_sigma = 1.
        params.sim_continue = true

        params.wf_sim_time = params.n * 2
        params.wf_p_fire = 0.06

        params.r_surveillance = 0.5

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

    simulate(params, draw = true, wait = true)

    #estimateExpectedUtility(params, N_min = 1000, N_max = 10000, RE_threshold = 0.01, verbose = 1)

    #evaluatePolicy(param_set, :back, N_min = 100, N_max = 1000, RE_threshold = 0.01)
end


