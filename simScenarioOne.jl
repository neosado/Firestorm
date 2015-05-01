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

using JSON
using Base.Test


macro neat(v)
    return :(round(signif($v, 4), 4))
end


type SimStat

    nlevel::Int64

    U::Vector{Float64}

    N::Vector{Int64}
    N_hit::Vector{Int64}

    n_total::Int64
    n_timestep::Int64


    function SimStat(max_level::Int64 = 0)

        self = new()

        self.nlevel = max_level + 1

        self.U = zeros(Float64, self.nlevel)

        self.N = zeros(Int64, self.nlevel)
        self.N_hit = zeros(Int64, self.nlevel)

        self.n_total = 0
        self.n_timestep = 0

        return self
    end
end


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

    params.r_crashed = -1000.
    params.r_dist = [1. -10000.; 2. -100.; 3. -20.]
    params.r_surveillance = 10.

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
    # debug
    ncollisions = 0

    for i = 1:N_max
        n += 1

        U = simulate(params)
        # debug
        #println(U)

        if U <= -10000
            ncollisions += 1
        end

        meanU += (U - meanU) / i
        ssU += U * U
        if i > 1
            var = (ssU - i * (meanU * meanU)) / ((i - 1) * i)
            RE = abs(sqrt(var) / meanU)

            if verbose >= 1 && i % 100 == 0
                #println("i: $i, mean: $meanU, RE: $RE")
                println("i: $i, mean: $meanU, RE: $RE, collisions: $ncollisions")
            end

            if i >= N_min && (isnan(RE) || (RE_threshold != 0 && RE < RE_threshold))
                break
            end
        end
    end

    if verbose >= 1
        #println("n: $n, mean: $meanU, RE: $RE")
        println("i: $n, mean: $meanU, RE: $RE, collisions: $ncollisions")
        println(C)
    end

    return meanU, RE, n
end


function simulate(params::ScenarioOneParams, L::Union(Vector{Float64}, Nothing), R::Union(Vector{Int64}, Int64), level::Int64, args::Union((ScenarioOne, ScenarioOneState, Int64, Float64), Nothing), sim_stat::SimStat; RE_threshold::Float64 = 0., verbose::Int64 = 0, nv::Int64 = 100, nv_interval::Int64 = 100)

    if typeof(R) == Int64
        N = R
    else
        N = R[level + 1]
    end

    n::Int64 = 0

    U_sum::Float64 = 0.
    U_mean::Float64 = 0.

    Y = zeros(Int64, N)
    prev_hit = 0

    LOG = {}

    for i = 1:N
        n += 1
        sim_stat.N[level+1] += 1

        if level == 0
            s1 = ScenarioOne(params)
            state = ScenarioOneState(s1)

            initTrajectories(s1)

            t = 0
            U = 0.
        else
            s1 = deepcopy(args[1])
            state = deepcopy(args[2])
            t = args[3] + 1
            U = args[4]
        end

        bHitLevel = false

        while !isEndState(s1, state, t)
            updateStatePartA(s1, state, t)

            r = getReward(s1, state, t)
            U += r

            updateStatePartB(s1, state, t)

            if state.uav_status == :collided
                @assert level == sim_stat.nlevel - 1
                sim_stat.N_hit[level+1] += 1
            end

            sim_stat.n_timestep += 1

            if L != nothing && level < sim_stat.nlevel - 1
                dist = abs(state.uav_loc[1] - state.aircraft_loc[1]) + abs(state.uav_loc[2] - state.aircraft_loc[2])

                if t <= s1.sim_comm_loss_duration && dist < L[level + 2]
                    sim_stat.N_hit[level+1] += 1
                    bHitLevel = true
                    break
                end
            end

            t += 1
        end

        if bHitLevel
            U = simulate(params, L, R, level + 1, (s1, state, t, U), sim_stat, verbose = verbose)
        else
            sim_stat.n_total += 1
        end

        U_sum += U
        sim_stat.U[level+1] += (U - sim_stat.U[level+1]) / sim_stat.N[level+1]

        if level == 0
            Y[i] = sim_stat.N_hit[sim_stat.nlevel] - prev_hit
            prev_hit = sim_stat.N_hit[sim_stat.nlevel]
        end

        if level == 0 && sim_stat.n_total >= nv
            U_mean = U_sum / n

            #println("i: ", sim_stat.n_total, ", mean: ", @neat(U_mean))
            #println("i: ", sim_stat.n_total, ", mean: ", U_mean)

            #if false
            #    for l = 1:sim_stat.nlevel
            #        n = sim_stat.N[l]
            #        n_hit = sim_stat.N_hit[l]
            #        #println("level: ", l, ", n: ", n, ", U: ", @neat(sim_stat.U[l]), ", p: ", @neat(n_hit/n), " ($n_hit/$n)")
            #        println("level: ", l, ", n: ", n, ", p: ", @neat(n_hit/n), " ($n_hit/$n)")
            #    end
            #end

            p = prod(sim_stat.N_hit ./ sim_stat.N)
            if typeof(R) == Int64
                std_ = sqrt(var(Y[1:i]) / i)
            else
                std_ = sqrt(var(Y[1:i]) / (i * prod(R[2:end].^2)))
            end
            RE = std_ / p

            if verbose >= 1
                println("i: ", sim_stat.n_total, ", s: ", sim_stat.n_timestep, ", p: ", p, ", RE: ", RE)
            end
            #println()

            push!(LOG, {sim_stat.n_total, sim_stat.n_timestep, p, RE})

            nv += nv_interval
        end
    end

    U_mean = U_sum / n

    if level == 0
        return U_mean, Y, LOG
    else
        return U_mean
    end
end


function estimateExpectedUtilityMS(params::ScenarioOneParams, L::Union(Nothing, Vector{Float64}), R::Union(Int64, Vector{Int64}); RE_threshold::Float64 = 0., verbose::Int64 = 0)

    if L == nothing
        max_level = 0
    else
        max_level = length(L) - 1
    end

    sim_stat = SimStat(max_level)

    U, Y, LOG = simulate(params, L, R, 0, nothing, sim_stat, RE_threshold = RE_threshold, verbose = verbose, nv = 100, nv_interval = 100)

    if verbose >= 1
        #println("n: ", sim_stat.n_total, ", mean: ", U)
        #println("n: ", sim_stat.n_total, ", mean: ", @neat(U))
        println()

        for l = 1:sim_stat.nlevel
            n = sim_stat.N[l]
            n_hit = sim_stat.N_hit[l]
            println("level: ", l, ", n: ", n, ", U: ", @neat(sim_stat.U[l]), ", p: ", @neat(n_hit/n), " ($n_hit/$n)")
        end
    end

    p = prod(sim_stat.N_hit ./ sim_stat.N)
    if typeof(R) == Int64
        std_ = sqrt(var(Y) / R[1])
    else
        std_ = sqrt(var(Y) / (R[1] * prod(R[2:end].^2)))
    end
    RE = std_ / p

    if verbose >= 1
        println("n: ", sim_stat.n_total, ", s: ", sim_stat.n_timestep, ", p: ", p, ", RE: ", RE)
    end

    push!(LOG, {sim_stat.n_total, sim_stat.n_timestep, p, RE})

    if verbose >= 1
        #if false
        #    n0 = R[1]
        #    n1 = R[2]
        #    p1 = sim_stat.N_hit[1] / sim_stat.N[1]
        #    p2 = sim_stat.N_hit[2] / sim_stat.N[2]
        #    var_p2 = 0.

        #    p = p1 * p2
        #    RE = sqrt(((1 - p1) / p1 + (1 - p2) / (n1 * p1 * p2) + (1 - 1 / n1) * var_p2 / (p1 * p2^2)) / n0)

        #    println("p: $p, RE: $RE")
        #end
    end

    return LOG
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

        params.wf_init_loc = [(6, 4), (7, 4), (5, 5), (6, 5), (7, 5), (4, 6), (5, 6), (6, 6), (7, 6), (5, 7), (6, 7), (6, 8)]
        params.wf_sim_time = 0
        params.wf_p_fire = 0.06

        params.r_surveillance = 10.

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

    params.uav_loc = (9, 4)
    params.uav_policy = :stay
    #params.sim_comm_loss_duration_sigma = 0.

    #simulate(params, draw = true, wait = true)

    #params.r_dist = [1. 0.; 2. -100.; 3. -20.]
    #estimateExpectedUtility(params, N_min = 1000, N_max = 1000000, RE_threshold = 0.01, verbose = 1)


    #L = nothing
    #R = 1000000

    L = [Inf, 3] # level 0, level 1, ...
    R = [700000, 10] # level 0, level 1, ...

    #L = [Inf, 3, 2] # level 0, level 1, ...
    #R = [100000, 7, 20] # level 0, level 1, ...

    estimateExpectedUtilityMS(params, L, R, RE_threshold = 0.01, verbose = 1)


    #evaluatePolicy("1.0", param_set, :back, N_min = 100, N_max = 1000, RE_threshold = 0.01)

    #if false
    #    s1 = ScenarioOne(params)
    #    state = ScenarioOneState(s1)
    #    draw = true
    #    bCopy = false

    #    initTrajectories(s1)

    #    if draw
    #        s1v = ScenarioOneVisualizer(wait = true)

    #        visInit(s1v, s1)
    #        visUpdate(s1v, s1)
    #        updateAnimation(s1v)
    #    end

    #    t = 0
    #    U = 0.

    #    s1_ = nothing
    #    state_ = nothing
    #    t_ = nothing
    #    U_ = U

    #    while false
    #        while !isEndState(s1, state, t)
    #            dist = abs(state.uav_loc[1] - state.aircraft_loc[1]) + abs(state.uav_loc[2] - state.aircraft_loc[2])
    #            if dist < 5 && bCopy == false
    #                bCopy = true
    #                break
    #            end

    #            updateStatePartA(s1, state, t)

    #            r = getReward(s1, state, t)
    #            U += r

    #            if draw
    #                visInit(s1v, s1, t)
    #                visUpdate(s1v, s1, t, state, r, U)
    #                updateAnimation(s1v, t)
    #            end

    #            updateStatePartB(s1, state, t)

    #            t += 1
    #        end

    #        if t > 30
    #            break
    #        else
    #            s1_ = deepcopy(s1)
    #            state_ = deepcopy(state)
    #            t_ = t
    #            U_ = U
    #            println("copy")
    #        end
    #    end

    #    if bCopy
    #        while true
    #            s1 = deepcopy(s1_)
    #            state = deepcopy(state_)
    #            t = t_
    #            U = U_

    #            if draw
    #                visInit(s1v, s1, t - 1)
    #                visUpdate(s1v, s1, t - 1, state, 0., U)
    #                updateAnimation(s1v, t - 1)
    #            end

    #            while !isEndState(s1, state, t)
    #                updateStatePartA(s1, state, t)

    #                r = getReward(s1, state, t)
    #                U += r

    #                if draw
    #                    visInit(s1v, s1, t)
    #                    visUpdate(s1v, s1, t, state, r, U)
    #                    updateAnimation(s1v, t)
    #                end

    #                updateStatePartB(s1, state, t)

    #                t += 1
    #            end
    #        end
    #    end
    #end
end


