# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 12/02/2014

using Wildfire_
using Firestorm_

using QMDP_
using FIB_
using UCT_
using POMCP_

using Util
using WildfireVisualizer_
using FirestormVisualizer_
using MCTSVisualizer_

using Iterators
using Base.Test


function sampleParticles(pm, b, nsample = 100000)

    B = FSState[]

    for n = 1:nsample
        rv = rand()

        sum_ = 0.
        for s in keys(b.belief)
            sum_ += b.belief[s]

            if rv < sum_
                push!(B, s)

                break
            end
        end
    end

    return FSBeliefParticles(B)
end


function beliefParticles2Vector(pm, B)

    count_ = Dict{FSState, Int64}()
    belief = Dict{FSState, Float64}()

    for s in pm.states
        count_[s] = 0
        belief[s] = 0.
    end

    sum_ = 0
    for s in B.particles
        count_[s] += 1
        sum_ += 1
    end
    sum_ = float(sum_)

    for s in B.particles
        belief[s] = count_[s] / sum_
    end

    return FSBeliefVector(belief)
end


function printBelief(pm, alg, b)

    if typeof(alg) == POMCP
        bv = beliefParticles2Vector(pm, b)
    else
        bv = b
    end

    for s in pm.states
        if s.Position == pm.uav_pos
            println(s.Position, ", ", vec(s.B), ": ", bv.belief[s])
        else
            @test bv.belief[s] == 0.
        end
    end
end


function getInitialState(pm::Firestorm)

    return FSState(pm.uav_pos, pm.wm.B)
end


function getInitialBelief(pm::Firestorm; bParticles::Bool = false)

    if bParticles
        B = FSState[]

        for B_array in product(repeated([false, true], pm.nrow * pm.ncol)...)
            push!(B, FSState(pm.uav_pos, reshape([B_array...], (pm.nrow, pm.ncol))))
        end

        return FSBeliefParticles(B)
    else
        belief = Dict{FSState, Float64}()

        sum_ = 0
        for s in pm.states
            if s.Position == pm.uav_pos
                belief[FSState(s.Position, s.B)] = 1.
                sum_ += 1
            else
                belief[FSState(s.Position, s.B)] = 0.
            end
        end

        for s in keys(belief)
            belief[s] /= sum_
        end

        return FSBeliefVector(belief)
    end
end


function test(pm, alg)

    if typeof(alg) == POMCP
        b = getInitialBelief(pm, bParticles = true)
    else
        b = getInitialBelief(pm)
    end
    printBelief(pm, alg, b)

    a_opt, Qv = selectAction(alg, pm, b)

    Qv__ = Float64[]
    for a in  pm.actions
        push!(Qv__, round(Qv[a], 2))
    end
    println("Qv: ", Qv__)
    println("action: ", a_opt.action)
end


function simulate_wildfire(wm, ts_max = 1; draw = false, wait = false)

    if draw
        wfv = WildfireVisualizer(wait = wait)

        visInit(wfv, wm)
        visUpdate(wfv, wm)
        updateAnimation(wfv)
    end

    for t = 1:ts_max
        wfNextState(wm)

        if draw
            visInit(wfv, wm)
            visUpdate(wfv, wm, t)
            updateAnimation(wfv)
        end
    end

    if draw
        saveAnimation(wfv)
        readline()
    end
end


function simulate(pm, alg; draw = true, wait = false)

    simulate_wildfire(pm.wm, max(pm.nrow, pm.ncol))

    if draw
        fsv = FirestormVisualizer(wait = wait)
    end

    s = getInitialState(pm)

    if typeof(alg) == POMCP
        b = getInitialBelief(pm, bParticles = true)
    else
        b = getInitialBelief(pm)
    end
    #printBelief(pm, alg, b)

    println("time: 0, s: ", s.Position)

    if draw
        visInit(fsv, pm, b)
        visUpdate(fsv, pm)
        updateAnimation(fsv)
    end

    R = 0.

    for i = 1:50
        #println("T: ", alg.T)
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
        #println("Q: ", alg.Q)
        #println("B: ", alg.B)
        #println()

        a, Qv = selectAction(alg, pm, b)

        #println("T: ", alg.T)
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
        #println("Q: ", alg.Q)
        #println("B: ", alg.B)
        #println()

        s_ = nextState(pm, s, a)

        o = observe(pm, s_, a)

        r = reward(pm, s, a, s_)
        R += r

        if typeof(alg) == POMCP
            b = updateBelief(pm, FSBeliefParticles(getParticles(alg, a, o)))
        else
            b = updateBelief(pm, b, a, o)
        end
        #printBelief(pm, alg, b)

        Qv__ = Float64[]
        for a__ in  pm.actions
            push!(Qv__, round(Qv[a__], 2))
        end
        println("time: ", i, ", s: ", s.Position, ", Qv: ", Qv__, ", a: ", a.action, ", o: ", o.observation, ", r: ", r, ", R: ", R, ", s_: ", s_.Position)

        updateInternalStates(pm, s, a, s_)

        s = s_

        if draw
            visInit(fsv, pm, b)
            visUpdate(fsv, pm, (i, a, o, r, R))
            updateAnimation(fsv)
        end

        if isEnd(pm, s)
            println("reached the terminal state")
            break
        end

        if typeof(alg) == UCT || typeof(alg) == POMCP
            reinitialize(alg, a, o)
        end
    end

    if draw
        saveAnimation(fsv, repeat = true)
    end
end


function default_policy(pm::Firestorm, s::FSState)

    a = pm.actions[rand(1:length(pm.actions))]

    while !isFeasible(pm::Firestorm, s, a)
        a = pm.actions[rand(1:length(pm.actions))]
    end

    return a
end


srand(uint(time()))

#pm = Firestorm(5, seed = rand(1:typemax(Int64)))
#pm = Firestorm(5, seed = rand(1:1024), p_fire = 0.12)
pm = Firestorm(3, seed = 837, p_fire = 0.25)

#alg = QMDP(pm, "firestorm_qmdp.pcy", gamma_ = 0.90, verbose = 1)
#alg = QMDP("firestorm_qmdp.pcy")
#alg = FIB(pm, "firestorm_fib.pcy", gamma_ = 0.90, verbose = 1)
#alg = FIB("firestorm_fib.pcy")

#alg = UCT(depth = 5, default_policy = default_policy, nloop_max = 10000, nloop_min = 10000, c = 20., gamma_ = 0.99, rgamma_ = 0.99, visualizer = MCTSVisualizer())
alg = POMCP(depth = 5, default_policy = default_policy, nloop_max = 10000, nloop_min = 10000, c = 20., gamma_ = 0.99, rgamma_ = 0.99, visualizer = MCTSVisualizer())

#wm = Wildfire(10, 10, p_fire = 0.12)
#simulate_wildfire(wm, 60, draw = true, wait = true)

#test(pm, alg)
simulate(pm, alg, draw = true, wait = false)


