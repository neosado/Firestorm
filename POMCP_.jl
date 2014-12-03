# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

# Partially Observable Monte-Carlo Planning
# D. Silver and J. Veness, "Monte-Carlo Planning in Large POMDPs," in Advances in Neural Information Processing Systems (NIPS), 2010.

module POMCP_

export POMCP, selectAction, reinitialize, initialize, getParticles


using MCTS_
using POMDP_
using Util

using Base.Test


import MCTS_.selectAction
import MCTS_.reinitialize
import MCTS_.initialize


type POMCP <: MCTS

    depth::Int64

    default_policy::Function

    T::Dict{History, Bool}
    N::Dict{(History, Action), Int64}
    Ns::Dict{History, Int64}
    Q::Dict{(History, Action), Float64}
    B::Dict{History, Vector{State}}

    nloop_max::Int64
    nloop_min::Int64
    eps::Float64

    c::Float64
    gamma_::Float64

    rgamma_::Float64

    reuse::Bool


    function POMCP(;depth::Int64 = 3, default_policy::Function = pi_0, nloop_max::Int64 = 10000, nloop_min::Int64 = 10000, eps::Float64 = 1.e-3, c::Float64 = 1., gamma_::Float64 = 0.9, rgamma_::Float64 = 0.9)

        self = new()

        self.depth = depth

        self.default_policy = default_policy

        self.T = Dict{History, Bool}()
        self.N = Dict{(History, Action), Int64}()
        self.Ns = Dict{History, Int64}()
        self.Q = Dict{(History, Action), Float64}()

        self.B = Dict{History, Vector{State}}()

        self.nloop_max = nloop_max
        self.nloop_min = nloop_min
        self.eps = eps

        self.c = c
        self.gamma_ = gamma_

        # rollout gamma
        self.rgamma_ = rgamma_

        self.reuse = false

        srand(int(time()))

        return self
    end
end


# \pi_0
pi_0(pm::POMDP, s::State) = pm.actions[rand(1:length(pm.actions))]


# s', o, r ~ G(s, a)
function Generative(pm::POMDP, s::State, a::Action)

    s_ = nextState(pm, s, a)
    @test s_ != nothing
    o = observe(pm, s_, a)
    if pm.reward_functype == :type2
        r = reward(pm, s, a)
    elseif pm.reward_functype == :type3
        r = reward(pm, s, a, s_)
    end

    return s_, o, r
end


function rollout(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64)

    if d == 0
        return 0
    end

    a = alg.default_policy(pm, s)

    s_, o, r = Generative(pm, s, a)

    if isEnd(pm, s_)
        return r
    end

    return r + alg.rgamma_ * rollout(alg, pm, s_, h, d - 1)
end


function simulate(alg::POMCP, pm::POMDP, s::State, h::History, d::Int64)

    if d == 0
        return 0
    end

    if !haskey(alg.T, h)
        #println("new node: ", h, " at level ", d)

        for a in pm.actions
            alg.N[(h, a)] = 0
            alg.Q[(h, a)] = 0
        end
        alg.Ns[h] = 0

        alg.T[h] = true

        ro = rollout(alg, pm, s, h, d)
        #println("rollout: ", ro)

        return ro
    end

    #println("found node: ", h, " at level ", d)

    Qv = Array(Float64, pm.nAction)
    for i = 1:pm.nAction
        a = pm.actions[i]

        if alg.N[(h, a)] == 0
            Qv[i] = Inf
        else
            Qv[i] = alg.Q[(h, a)] + alg.c * sqrt(log(alg.Ns[h]) / alg.N[(h, a)])
        end
    end

    a = pm.actions[argmax(Qv)]

    s_, o, r = Generative(pm, s, a)

    #println("Qv: ", round(Qv, 2), ", (a, o): {", a, ", ", o, "}, s_: ", s_, ", r: ", r)

    if isEnd(pm, s_)
        q = r
    else
        q = r + alg.gamma_ * simulate(alg, pm, s_, History([h.history, a, o]), d - 1)
    end

    if !haskey(alg.B, h)
        alg.B[h] = [s]
    else
        push!(alg.B[h], s)
    end

    alg.N[(h, a)] += 1
    alg.Ns[h] += 1
    alg.Q[(h, a)] += (q - alg.Q[(h, a)]) / alg.N[(h, a)]

    return q
end


function selectAction(alg::POMCP, pm::POMDP, b::Belief)

    h = History()

    if !alg.reuse
        initialize(alg)
    end

    Qv = Dict{Action, Float64}()
    for a in pm.actions
        Qv[a] = 0.
    end

    for i = 1:alg.nloop_max
        #println("iteration: ", i)

        s = sampleBelief(pm, b)

        simulate(alg, pm, s, h, alg.depth)

        #println("h: ", h)
        #println("T: ", alg.T)
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
        #println("Q: ", alg.Q)
        #println("B: ", alg.B)
        #println()

        res = 0.
        for a in pm.actions
            Qv_prev = Qv[a]
            Qv[a] = alg.Q[(h, a)]
            res += (Qv[a] - Qv_prev)^2
        end

        if i > alg.nloop_min &&  sqrt(res) < alg.eps
            break
        end
    end

    Qv_max = -Inf
    for a in pm.actions
        Qv[a] = alg.Q[(h, a)]

        if Qv[a] > Qv_max
            Qv_max = Qv[a]
        end
    end

    actions = Action[]
    for a in pm.actions
        if Qv[a] == Qv_max
            push!(actions, a)
        end
    end
    action = actions[rand(1:length(actions))]

    return action, Qv
end


function reinitialize(alg::POMCP, a::Action, o::Observation)

    T_new = Dict{History, Bool}()
    N_new = Dict{(History, Action), Int64}()
    Ns_new = Dict{History, Int64}()
    Q_new = Dict{(History, Action), Float64}()
    B_new = Dict{History, Vector{State}}()

    for h in keys(alg.T)
        if length(h.history) > 0 && h.history[1] == a && h.history[2] == o
            T_new[History(h.history[3:end])] = alg.T[h]
            Ns_new[History(h.history[3:end])] = alg.Ns[h]
        end
    end

    for h in keys(alg.B)
        if length(h.history) > 0 && h.history[1] == a && h.history[2] == o
            B_new[History(h.history[3:end])] = alg.B[h]
        end
    end

    for key in keys(alg.N)
        h, action = key
        if length(h.history) > 0 && h.history[1] == a && h.history[2] == o
            N_new[(History(h.history[3:end]), action)] = alg.N[key]
            Q_new[(History(h.history[3:end]), action)] = alg.Q[key]
        end
    end

    alg.T = T_new
    alg.N = N_new
    alg.Ns = Ns_new
    alg.Q = Q_new
    alg.B = B_new

    alg.reuse = true
end


function initialize(alg::POMCP)

    alg.T = Dict{History, Bool}()
    alg.N = Dict{(History, Action), Int64}()
    alg.Ns = Dict{History, Int64}()
    alg.Q = Dict{(History, Action), Float64}()
    alg.B = Dict{History, Vector{State}}()

    alg.reuse = false
end


function getParticles(alg::POMCP, a::Action, o::Observation)

    return alg.B[History([a, o])]
end


end


