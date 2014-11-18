# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

using BabyCrying_
using UCT_
using POMCP_


function sampleParticles(pm, b, nsample = 1000000)

    B = BCState[]

    for n = 1:nsample
        rv = rand()

        sum_ = 0.
        for i = 1:length(b.belief)
            sum_ += b.belief[i]

            if rv < sum_
                push!(B, pm.states[i])

                break
            end
        end
    end

    return BCBeliefParticles(B)
end


function beliefParticles2Vector(pm, B)

    map_ = Dict()
    for i = 1:pm.nState
        state = pm.states[i]
        map_[state] = i
    end

    count = zeros(pm.nState)
    for state in B.particles
        count[map_[state]] += 1
    end

    belief = count / sum(count)

    return BCBeliefVector(belief)
end


function test(pm, alg)

    prob_hungry = 0.5
    b = BCBeliefVector([1 - prob_hungry, prob_hungry])

    if typeof(alg) == POMCP
        B = sampleParticles(pm, b)
        #println(beliefParticles2Vector(pm, B))

        a_opt, Qv = selectAction(alg, pm, B)

        b = beliefParticles2Vector(pm, BCBeliefParticles(alg.B[History()]))
        #println(b)
    else
        a_opt, Qv = selectAction(alg, pm, b)
    end

    println("p_hungry: ", prob_hungry)
    println("Qv: ", round(Qv, 2))
    println("action: ", a_opt)
end


function validate(pm, alg)

    for p = [0:0.1:1]   # probability of hungry
        b = BCBeliefVector([1 - p, p])

        if typeof(alg) == POMCP
            B = sampleParticles(pm, b)
            #println(beliefParticles2Vector(pm, B))

            a, Qv = selectAction(alg, pm, B)

            #b = beliefParticles2Vector(pm, BCBeliefParticles(alg.B[History()]))
            #println(b)
        else
            a, Qv = selectAction(alg, pm, b)
        end

        #println("b: ", b')
        println("p_hungry: ", p)
        println("Qv: ", round(Qv, 2))
        println("action: ", a)
        println()
    end
end


function simulate_scenario(pm, alg)

    actions = [BCAction(:notfeed), BCAction(:feed), BCAction(:notfeed), BCAction(:notfeed), BCAction(:notfeed)]
    observations = [BCObservation(:crying), BCObservation(:notcrying), BCObservation(:notcrying), BCObservation(:notcrying), BCObservation(:crying)]

    b = BCBeliefVector([0.5, 0.5])

    if typeof(alg) == POMCP
        B = sampleParticles(pm, b)
        b = beliefParticles2Vector(pm, B)
    end

    println(b)

    for i = 1:length(actions)
        #println("T: ", alg.T)
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
        #println("Q: ", alg.Q)
        #println("B: ", alg.B)
        #println()

        if typeof(alg) == POMCP
            selectAction(alg, pm, B)
        end

        #println("T: ", alg.T)
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
        #println("Q: ", alg.Q)
        #println("B: ", alg.B)
        #println()

        a = actions[i]
        o = observations[i]

        if typeof(alg) == POMCP
            B = updateBelief(pm, BCBeliefParticles(getParticles(alg, a, o)))
            b = beliefParticles2Vector(pm, B)
        else
            b = updateBelief(pm, b, a, o)
        end

        println(b)

        reinitialize(alg, a, o)
    end

end


function simulate(pm, alg)

    s = BCState(:nothungry)
    b = BCBeliefVector([0.5, 0.5])

    if typeof(alg) == POMCP
        B = sampleParticles(pm, b)
        b = beliefParticles2Vector(pm, B)
    end

    R = 0

    println("time: 0, s: ", s, ", b: ", b)

    for i = 1:100
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

        o = observe(pm, s_, a)

        r = reward(pm, s, a)
        R += r

        s = s_

        println("time: ", i, ", s: ", s, ", Qv: ", round(Qv, 2), ", a: ", a, ", r: ", r, ", o: ", o, ", b: ", b, ", R: ", R, ", s_: ", s_)

        if typeof(alg) == POMCP
            B = updateBelief(pm, BCBeliefParticles(getParticles(alg, a, o)))
            b = beliefParticles2Vector(pm, B)
        else
            b = updateBelief(pm, b, a, o)
        end

        reinitialize(alg, a, o)
    end
end


srand(uint(time()))

pm = BabyCrying()

#alg = UCT()
alg = POMCP()

#test(pm, alg)
#validate(pm, alg)
simulate_scenario(pm, alg)
#simulate(pm, alg)


