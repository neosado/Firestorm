# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/13/2015

using Wildfire_
using WildfireVisualizer_

using ScenarioOne_
using ScenarioOneVisualizer_

using PyPlot
using Base.Test


function generateParams(setnum::Int64 = 1)

    params = ScenarioOneParams()

    if setnum == 1
        params.n = 11

        params.wf_init_loc = (int(params.n / 2), int(params.n / 2))
        params.wf_p_fire = 0.12

        params.p_uav_landing_crash = 0.01

        params.r_crashed = -15
        params.r_dist = [1 -100; 2 -20; 3 -10]

        params.uav_loc = (4, 5)
        params.uav_base_loc = (11, 8)
        params.uav_velocity = 0.3
        params.uav_policy = :back

        params.aircraft_start_loc = (3, 11)
        params.aircraft_end_loc = (10, 1)
        params.aircraft_control_points = [(6.5, 6.)]
        params.aircraft_velocity = 1.
        params.aircraft_traj_uncertainty = 0.

    elseif setnum == 2
        params.n = 11

        params.wf_init_loc = (int(params.n / 2), int(params.n / 2))
        params.wf_p_fire = 0.12

        params.p_uav_landing_crash = 0.01

        params.r_crashed = -15
        params.r_dist = [1 -100; 2 -20; 3 -10]

        params.uav_loc = (4, 5)
        params.uav_base_loc = (11, 8)
        params.uav_velocity = 0.3
        params.uav_policy = :back

        params.aircraft_start_loc = (4, 11)
        params.aircraft_end_loc = (7, 11)
        params.aircraft_control_points = [(-2., 1.), (13., 1.)]
        params.aircraft_velocity = 1.
        params.aircraft_traj_uncertainty = 0.

    else
        error("invalid parameter set")
    end

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
    U = 0

    while !isEndState(s1, state)
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


function evaluatePolicy(param_set_num::Int64, policy::Symbol, uncertainty::Float64; N_min::Int = 0, N_max::Int = 1000, RE_threshold::Float64 = 0., bParallel::Bool = false)

    params = generateParams(param_set_num)

    params.uav_policy = policy
    params.aircraft_traj_uncertainty = uncertainty


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

    param_set = 1

    params = generateParams(param_set)

    params.uav_loc = (4, 5)
    params.uav_policy = :back
    params.aircraft_traj_uncertainty = 1.

    simulate(params, draw = true, wait = true)

    #estimateExpectedUtility(params, N_min = 1000, N_max = 10000, RE_threshold = 0.01, verbose = 1)

    #evaluatePolicy(param_set, :back, 0., N_min = 100, N_max = 1000, RE_threshold = 0.01)
end


