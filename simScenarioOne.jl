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
    U2::Float64

    N::Vector{Int64}
    N_hit::Vector{Int64}

    n_total::Int64
    n_timestep::Int64


    function SimStat(max_level::Int64 = 0)

        self = new()

        self.nlevel = max_level + 1

        self.U = zeros(Float64, self.nlevel)
        self.U2 = 0.

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
    bCollided = false

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

        if state.uav_status == :collided
            bCollided = true
        end

        t += 1
    end

    if draw
        saveAnimation(s1v, repeat = true)
    end

    return U, bCollided
end


function estimateExpectedUtility(params::ScenarioOneParams; N_min::Int = 0, N_max::Int = 1000, RE_threshold::Float64 = 0., MS::Bool = false, verbose::Int64 = 0)

    meanU = 0.
    ssU = 0.
    RE = 0.

    meanU1 = 0.

    n = 0
    ncollisions = 0

    for i = 1:N_max
        n += 1

        U, bCollided = simulate(params)
        # debug
        #println(U)

        if bCollided
            ncollisions += 1
        else
            meanU1 += (U - meanU1) / (i - ncollisions)
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
    end

    if MS
        meanU_ = NaN
        RE_ = NaN
        n_ = 0
        p = ncollisions / n

        if ncollisions < int64(n / 100)
            L = [Inf, 3]
            #R = [max(int64(n / 10), 10000), 10]
            R = [n, 10]

            sim_stat = SimStat(length(L) - 1)

            U_, p, RE_, Y, LOG = simulate(params, L, R, 0, nothing, sim_stat, RE_threshold = RE_threshold, verbose = verbose, nv = 100, nv_interval = 100)

            if verbose >= 1
                for l = 1:sim_stat.nlevel
                    n_ = sim_stat.N[l]
                    n_hit = sim_stat.N_hit[l]
                    println("level: ", l, ", n: ", n_, ", U: ", @neat(sim_stat.U[l]), ", p: ", @neat(n_hit/n), " ($n_hit/$n_)")
                end
            end

            meanU_ = (1 - p) * meanU1 + p * sim_stat.U2
            RE_ = NaN
            n_ = int64(sim_stat.n_timestep / params.sim_time)

            if verbose >= 1
                println("U: ", meanU, ", ncoll.: ", ncollisions, ", U1: ", meanU1, ", U2: ", sim_stat.U2, ", p: ", p, ", U_: ", meanU_)
            end
        end

        return meanU, RE, n, meanU_, RE_, n_, p

    else
        return meanU, RE, n

    end
end


function simulate(params::ScenarioOneParams, L::Union(Vector{Float64}, Nothing), R::Union(Vector{Int64}, Int64), level::Int64, args::Union((ScenarioOne, ScenarioOneState, Int64, Float64), Nothing), sim_stat::SimStat, T = nothing; RE_threshold::Float64 = 0., verbose::Int64 = 0, nv::Int64 = 100, nv_interval::Int64 = 100)

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

            if t != 0
                sim_stat.n_timestep += 1
            end

            if L != nothing && level < sim_stat.nlevel - 1
                if state.uav_status != :left && state.aircraft_status != :left
                    dist = abs(state.uav_loc[1] - state.aircraft_loc[1]) + abs(state.uav_loc[2] - state.aircraft_loc[2])

                    if t <= s1.sim_comm_loss_duration && dist < L[level + 2]
                        sim_stat.N_hit[level+1] += 1
                        bHitLevel = true
                        break
                    end
                end
            end

            t += 1
        end

        if bHitLevel
            if T == nothing
                T_ = nothing
            else
                T_ = {}
                push!(T, (1, T_))
            end

            U = simulate(params, L, R, level + 1, (s1, state, t, U), sim_stat, T_, verbose = verbose)
        else
            if T != nothing
                if state.uav_status == :collided
                    push!(T, (1, {}))
                else
                    push!(T, (0, {}))
                end
            end

            sim_stat.n_total += 1
        end

        U_sum += U
        sim_stat.U[level+1] += (U - sim_stat.U[level+1]) / sim_stat.N[level+1]

        if state.uav_status == :collided
            sim_stat.U2 += (U - sim_stat.U2) / sim_stat.N_hit[level+1]
        end

        if level == 0
            Y[i] = sim_stat.N_hit[sim_stat.nlevel] - prev_hit
            prev_hit = sim_stat.N_hit[sim_stat.nlevel]
        end

        if level == 0 && sim_stat.n_total >= nv
            #U_mean = U_sum / n

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

            if sim_stat.N_hit[sim_stat.nlevel] == 0
                p = 0.
            else
                p = prod(sim_stat.N_hit ./ sim_stat.N)
            end
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

            if RE_threshold != 0 && RE < RE_threshold
                break
            end
        end
    end

    U_mean = U_sum / n

    if level == 0
        if sim_stat.N_hit[sim_stat.nlevel] == 0
            p = 0.
        else
            p = prod(sim_stat.N_hit ./ sim_stat.N)
        end
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

        return U_mean, p, RE, Y, LOG
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
    T = {}

    U, p, RE, Y, LOG = simulate(params, L, R, 0, nothing, sim_stat, T, RE_threshold = RE_threshold, verbose = verbose, nv = 100, nv_interval = 100)

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

    #if verbose >= 1
    #    n0 = R[1]
    #    n1 = R[2]
    #    p1 = sim_stat.N_hit[1] / sim_stat.N[1]
    #    p2 = sim_stat.N_hit[2] / sim_stat.N[2]

    #    S = {}
    #    for i = 1:n1
    #        push!(S, Int64[])
    #    end

    #    for (t1, T1) in T
    #        i = 1
    #        for (t2, T2) in T1
    #            push!(S[i], t2)
    #            i += 1
    #        end
    #    end

    #    S_ = zeros(n1)
    #    for i = 1:n1
    #        S_[i] = mean(S[i])
    #    end

    #    var_p2 = var(S_)
    #    println("var(p2): ", var_p2, ", c: ", var_p2 / p2^2)

    #    p = p1 * p2
    #    RE = sqrt(((1 - p1) / p1 + (1 - p2) / (n1 * p1 * p2) + (1 - 1 / n1) * var_p2 / (p1 * p2^2)) / n0)

    #    println("p: $p, RE: $RE")
    #end

    return LOG
end


function evaluatePolicy(version::ASCIIString, param_set_num::Int64, policy::Symbol; sim_comm_loss_duration_mu::Union(Float64, Nothing) = nothing, sim_comm_loss_duration_sigma::Float64 = 0., r_surveillance::Union(Float64, Nothing) = nothing, uav_surveillance_pattern::Union(Symbol, Nothing) = nothing, aircraft_traj_uncertainty::Union(Float64, Nothing) = nothing, N_min::Int = 0, N_max::Int = 1000, RE_threshold::Float64 = 0., MS::Bool = false, bParallel::Bool = false)

    params = generateParams(param_set_num)

    params.uav_policy = policy

    if version == "0.1"
        @assert aircraft_traj_uncertainty != nothing

        params.sim_comm_loss_duration_mu = 10.

        params.wf_init_loc = [(6, 4), (7, 4), (5, 5), (6, 5), (7, 5), (4, 6), (5, 6), (6, 6), (7, 6), (5, 7), (6, 7), (6, 8)]
        params.wf_sim_time = 0
        params.wf_p_fire = 0.06

        params.aircraft_traj_uncertainty = 1.

    elseif version == "0.2"
        @assert sim_comm_loss_duration_mu != nothing

        params.sim_comm_loss_duration_mu = sim_comm_loss_duration_mu
        params.sim_comm_loss_duration_sigma = sim_comm_loss_duration_sigma

        params.wf_init_loc = [(6, 4), (7, 4), (5, 5), (6, 5), (7, 5), (4, 6), (5, 6), (6, 6), (7, 6), (5, 7), (6, 7), (6, 8)]
        params.wf_sim_time = 0
        params.wf_p_fire = 0.06

        params.aircraft_traj_uncertainty = 0.
        params.aircraft_traj_adaptive = true
        params.aircraft_operation_time_limit = 30

    elseif version == "1.0"
        @assert uav_surveillance_pattern != nothing

        params.sim_time = 30
        params.sim_comm_loss_duration_mu = 10.
        params.sim_comm_loss_duration_sigma = 1.
        params.sim_continue = true

        params.wf_init_loc = [(6, 4), (7, 4), (5, 5), (6, 5), (7, 5), (4, 6), (5, 6), (6, 6), (7, 6), (5, 7), (6, 7), (6, 8)]
        params.wf_sim_time = 0
        params.wf_p_fire = 0.06

        params.r_surveillance = r_surveillance

        params.uav_surveillance_pattern = uav_surveillance_pattern
        params.uav_cas = :lower

        params.aircraft_traj_uncertainty = 0.
        params.aircraft_traj_adaptive = true
        params.aircraft_operation_time_limit = 0

    end


    U = zeros(params.n, params.n)
    RE = zeros(params.n, params.n)
    N = zeros(Int64, params.n, params.n)

    if MS
        U_ = zeros(params.n, params.n)
        RE_ = zeros(params.n, params.n)
        N_ = zeros(Int64, params.n, params.n)
        P = zeros(params.n, params.n)
    end

    if bParallel == false
        for i = 1:params.n
            for j = 1:params.n
                params.uav_loc = (i, j)

                result = estimateExpectedUtility(params, N_min = N_min, N_max = N_max, RE_threshold = RE_threshold, MS = MS)

                U[i, j] = result[1]
                RE[i, j] = result[2]
                N[i, j] = result[3]

                #println("($i, $j), mean: ", int(U[i, j]), ", RE: ", signif(RE[i, j], 4), ", N: ", N[i, j])

                if MS
                    U_[i, j] = result[4]
                    RE_[i, j] = result[5]
                    N_[i, j] = result[6]
                    P[i, j] = result[7]
                end
            end
        end

    else
        lst = (Int64, Int64)[]
        for j = 1:params.n
            for i = 1:params.n
                push!(lst, (i, j))
            end
        end

        results = pmap(x -> (params.uav_loc = x; estimateExpectedUtility(params, N_min = N_min, N_max = N_max, RE_threshold = RE_threshold, MS = MS)), lst)

        k = 1
        for result in results
            U[k] = result[1]
            RE[k] = result[2]
            N[k] = result[3]

            if MS
                U_[k] = result[4]
                RE_[k] = result[5]
                N_[k] = result[6]
                P[k] = result[7]
            end

            k += 1
        end
    end

    # debug
    #println(params)
    #println(int(U))
    #println(round(RE, 4))
    #println(N)

    if MS
        return params, U, RE, N, U_, RE_, N_, P
    else
        return params, U, RE, N
    end
end


if false
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
        params.sim_comm_loss_duration_mu = 15.
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

    params.uav_loc = (9, 4)
    params.uav_policy = :stay
    #params.sim_comm_loss_duration_sigma = 0.

    #simulate(params, draw = true, wait = true)

    #params.r_dist = [1. 0.; 2. -100.; 3. -20.]
    #result = estimateExpectedUtility(params, N_min = 1000, N_max = 10000, RE_threshold = 0.01, MS = true, verbose = 1)
    #println(result)


    #L = nothing
    #R = 1000000

    #L = [Inf, 3] # level 0, level 1, ...
    #R = [700000, 10] # level 0, level 1, ...

    #L = [Inf, 3, 2] # level 0, level 1, ...
    #R = [100000, 7, 20] # level 0, level 1, ...

    #L = nothing
    #R = 1000
    #L = [Inf, 3]
    #R = [100000, 10]

    #estimateExpectedUtilityMS(params, L, R, RE_threshold = 0.01, verbose = 1)


    #evaluatePolicy("1.0", param_set, params.uav_policy, r_surveillance = params.r_surveillance, uav_surveillance_pattern = params.uav_surveillance_pattern, N_min = 100, N_max = 1000, RE_threshold = 0.01, MS = true)

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


