# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

# Upper Confidence Bound for Trees
# L. Kocsis and C. Szepesvari, "Bandit Based Monte-Carlo Planning," in European Conference on Machine Learning (ECML), 2006.

module UCT_

export UCT, selectAction, reinitialize, initialize


using MCTS_
using POMDP_
using Util

using Base.Test
using JSON


import MCTS_.selectAction
import MCTS_.reinitialize
import MCTS_.initialize


type UCT <: MCTS

    depth::Int64

    default_policy::Function

    T::Dict{History, Bool}
    N::Dict{(History, Action), Int64}
    Ns::Dict{History, Int64}
    Q::Dict{(History, Action), Float64}

    nloop_max::Int64
    nloop_min::Int64
    eps::Float64

    c::Float64
    gamma_::Float64

    rgamma_::Float64

    reuse::Bool


    function UCT(;depth::Int64 = 3, default_policy::Function = pi_0, nloop_max::Int64 = 10000, nloop_min::Int64 = 10000, eps::Float64 = 1.e-3, c::Float64 = 1., gamma_::Float64 = 0.9, rgamma_::Float64 = 0.9)

        self = new()

        self.depth = depth

        self.default_policy = default_policy

        self.T = Dict{History, Bool}()
        self.N = Dict{(History, Action), Int64}()
        self.Ns = Dict{History, Int64}()
        self.Q = Dict{(History, Action), Float64}()

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


function rollout(alg::UCT, pm::POMDP, s::State, h::History, d::Int64)

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


function simulate(alg::UCT, pm::POMDP, s::State, h::History, d::Int64; Troot = nothing)

    if Troot != nothing
        st = string(s)

        if !haskey(Troot, st)
            Troot[st] = Dict{ASCIIString, Any}()
            Troot[st]["actions"] = Dict{ASCIIString, Any}()
            Troot[st]["N"] = 1
        else
            Troot[st]["N"] += 1
        end
    end

    if d == 0 || isEnd(pm, s)
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

    if Troot == nothing
        q = r + alg.gamma_ * simulate(alg, pm, s_, History([h.history, a, o]), d - 1)
    else
        act = string(a.action)
        obs = string(o.observation)
        ns = string(s_)

        if !haskey(Troot[st]["actions"], act)
            Troot[st]["actions"][act] = Dict{ASCIIString, Any}()
            Troot[st]["actions"][act]["observations"] = Dict{ASCIIString, Any}()
            Troot[st]["actions"][act]["N"] = 0
            Troot[st]["actions"][act]["r"] = 0.
        end

        if !haskey(Troot[st]["actions"][act]["observations"], obs)
            Troot[st]["actions"][act]["observations"][obs] = Dict{ASCIIString, Any}()
            Troot[st]["actions"][act]["observations"][obs]["states"] = Dict{ASCIIString, Any}()
        end

        q = r + alg.gamma_ * simulate(alg, pm, s_, History([h.history, a, o]), d - 1, Troot = Troot[st]["actions"][act]["observations"][obs]["states"])
    end

    alg.N[(h, a)] += 1
    alg.Ns[h] += 1
    alg.Q[(h, a)] += (q - alg.Q[(h, a)]) / alg.N[(h, a)]

    if Troot != nothing
        Troot[st]["actions"][act]["N"] += 1
        Troot[st]["actions"][act]["r"] += (r - Troot[st]["actions"][act]["r"]) / Troot[st]["actions"][act]["N"]
    end

    return q
end


function selectAction(alg::UCT, pm::POMDP, b::Belief)

    h = History()
    Tvis = Dict{ASCIIString, Any}()

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

        simulate(alg, pm, s, h, alg.depth, Troot = Tvis)

        #println("h: ", h)
        #println("T: ", alg.T)
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
        #println("Q: ", alg.Q)
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

    # XXX Debug
    dumpTree("mcts.json", Tvis)

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


function dumpTree(output_file::ASCIIString, Tvis::Dict{ASCIIString, Any})

    function process(Tin, Tout, level; r_prev = 0.)

        if rem(level, 3) == 0
            for (state, node) in Tin
                node_ = Dict{ASCIIString, Any}()
                node_["state"] = state
                node_["N"] = node["N"]
                node_["actions"] = Dict{ASCIIString, Any}[]

                push!(Tout, node_)

                process(node["actions"], node_["actions"], level + 1, r_prev = r_prev)
            end
        elseif rem(level, 3) == 1
            for (action, node) in Tin
                node_ = Dict{ASCIIString, Any}()
                node_["action"] = action
                node_["N"] = node["N"]
                node_["r"] = node["r"]
                node_["R"] = r_prev + node["r"]
                node_["observations"] = Dict{ASCIIString, Any}[]

                push!(Tout, node_)

                process(node["observations"], node_["observations"], level + 1, r_prev = node_["R"])
            end
        elseif rem(level, 3) == 2
            for (observation, node) in Tin
                node_ = Dict{ASCIIString, Any}()
                node_["observation"] = observation
                node_["states"] = Dict{ASCIIString, Any}[]

                push!(Tout, node_)

                process(node["states"], node_["states"], level + 1, r_prev = r_prev)
            end
        end
    end

    Tout = Dict{ASCIIString, Any}()
    Tout["name"] = "root"
    Tout["states"] = Dict{ASCIIString, Any}[]

    process(Tvis, Tout["states"], 0)

    f = open(output_file, "w")
    JSON.print(f, Tout, 2)
    close(f)

    return Tout
end


function reinitialize(alg::UCT, a::Action, o::Observation)

    T_new = Dict{History, Bool}()
    N_new = Dict{(History, Action), Int64}()
    Ns_new = Dict{History, Int64}()
    Q_new = Dict{(History, Action), Float64}()

    for h in keys(alg.T)
        if length(h.history) > 0 && h.history[1] == a && h.history[2] == o
            T_new[History(h.history[3:end])] = alg.T[h]
            Ns_new[History(h.history[3:end])] = alg.Ns[h]
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

    alg.reuse = true
end


function initialize(alg::UCT)

    alg.T = Dict{History, Bool}()
    alg.N = Dict{(History, Action), Int64}()
    alg.Ns = Dict{History, Int64}()
    alg.Q = Dict{(History, Action), Float64}()

    alg.reuse = false
end


end


