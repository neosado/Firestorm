# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

using RockSample_
using UCT_
using POMCP_
using Util
using RockSampleVisualizer_

using Iterators
using Base.Test


function sampleParticles(pm, b, nsample = 100000)

    B = RSState[]

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

    return RSBeliefParticles(B)
end


function beliefParticles2Vector(pm, B)

    count_ = Dict{RSState, Int64}()
    belief = Dict{RSState, Float64}()

    for col = 1:pm.n
        for row = 1:pm.n 
            for rock_types in product(repeated([:good, :bad], pm.k)...)
                count_[RSState((row, col), [rock_types...])] = 0
            end
        end
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

    return RSBeliefVector(belief)
end


function getInitialState(pm::RockSample)

    rock_types = Array(Symbol, pm.k)

    for (rock, rock_type) in pm.rock_types
        rock_index = rock2ind(rock)
        rock_types[rock_index] = rock_type
    end

    return RSState(pm.rover_pos, rock_types)
end


function getInitialBelief(pm::RockSample)

    belief = Dict{RSState, Float64}()

    sum_ = 0
    for col = 1:pm.n
        for row = 1:pm.n 
            for rock_types in product(repeated([:good, :bad], pm.k)...)
                if pm.rover_pos == (row, col)
                    belief[RSState((row, col), [rock_types...])] = 1.
                    sum_ += 1
                else
                    belief[RSState((row, col), [rock_types...])] = 0.
                end
            end
        end
    end

    for s in keys(belief)
        belief[s] /= sum_
    end

    return RSBeliefVector(belief)
end


function test(pm, alg)

    b = getInitialBelief(pm)

    if typeof(alg) == POMCP
        B = sampleParticles(pm, b)

        a_opt, Qv = selectAction(alg, pm, B)
    else
        a_opt, Qv = selectAction(alg, pm, b)
    end

    println("Qv: ", round(Qv, 2))
    println("action: ", a_opt)
end


function simulate(pm, alg)

    rsv = RockSampleVisualizer(wait = false)
    dispWorld(rsv, pm)
    dispInitInfo(rsv, pm)
    updateAnimation(rsv)

    s = getInitialState(pm)
    b = getInitialBelief(pm)

    if typeof(alg) == POMCP
        B = sampleParticles(pm, b)
    end

    R = 0.

    println("time: 0, s: ", s)

    for i = 1:50
        #println("T: ", alg.T)
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
        #println("Q: ", alg.Q)
        #println("B: ", alg.B)
        #println()

        if typeof(alg) == POMCP
            a, Qv = selectAction(alg, pm, B)
        else
            a, Qv = selectAction(alg, pm, b)
        end

        #println("T: ", alg.T)
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
        #println("Q: ", alg.Q)
        #println("B: ", alg.B)
        #println()

        s_ = nextState(pm, s, a)
        @test s_ != nothing

        o = observe(pm, s_, a)

        r = reward(pm, s, a)
        R += r

        updateInternalStates(pm, s, a, s_)

        s = s_

        println("time: ", i, ", s: ", s, ", Qv: ", round(Qv, 2), ", a: ", a, ", o: ", o, ", r: ", r, ", R: ", R)

        dispWorld(rsv, pm)
        dispSimInfo(rsv, pm, (i, a, o, r, R))
        updateAnimation(rsv)

        if isEnd(pm, s)
            println("reached the terminal state")
            break
        end

        if typeof(alg) == POMCP
            B = updateBelief(pm, RSBeliefParticles(getParticles(alg, a, o)))
        else
            b = updateBelief(pm, b, a, o)
        end

        reinitialize(alg, a, o)
    end

    saveAnimation(rsv, repeat = true)
end


function isFeasible(pm::RockSample, s::RSState, a::RSAction)

    row, col = s.Position

    if a.action == :north && row == 1
        return false
    elseif a.action == :south && row == pm.n
        return false
    elseif a.action == :west && col == 1
        return false
    end

    return true
end

function default_policy(pm::RockSample, s::RSState)

    a = pm.actions[rand(1:length(pm.actions))]

    while !isFeasible(pm::RockSample, s, a)
        a = pm.actions[rand(1:length(pm.actions))]
    end

    return a
end


srand(uint(time()))

pm = RockSample(5, 5, seed = rand(1:typemax(Int64)))

#alg = UCT(depth = 5, default_policy = default_policy, nloop_max = 10000, nloop_min = 10000, c = 10.)
alg = POMCP(depth = 5, default_policy = default_policy, nloop_max = 10000, nloop_min = 10000, c = 10.)

#test(pm, alg)
simulate(pm, alg)


