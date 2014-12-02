# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

using BabyCrying_

using QMDP_
using FIB_
using UCT_
using POMCP_


function sampleParticles(pm, b, nsample = 100000)

    B = BCState[]

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

    return BCBeliefParticles(B)
end


function beliefParticles2Vector(pm, B)

    count_ = Dict{BCState, Int64}()
    belief = Dict{BCState, Float64}()

    for s in pm.states
        count_[s] = 0
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

    return BCBeliefVector(belief)
end


function test(pm, alg)

    prob_hungry = 0.5
    b = BCBeliefVector([BCState(:nothungry) => 1. - prob_hungry, BCState(:hungry) => prob_hungry])

    if typeof(alg) == POMCP
        B = sampleParticles(pm, b)
        #b = beliefParticles2Vector(pm, B)
        #println("b: ", float(map((x) -> x[2], b.belief)))

        a_opt, Qv = selectAction(alg, pm, B)

        b = beliefParticles2Vector(pm, BCBeliefParticles(alg.B[History()]))
        #println("b: ", float(map((x) -> x[2], b.belief)))
    else
        a_opt, Qv = selectAction(alg, pm, b)
    end

    println("p_hungry: ", prob_hungry)
    println("Qv: ", float(map((x) -> round(x[2], 2), Qv)))
    println("action: ", a_opt.action)
end


function validate(pm, alg)

    for p = [0:0.1:1]   # probability of hungry
        b = BCBeliefVector([BCState(:nothungry) => 1. - p, BCState(:hungry) => p])

        if typeof(alg) == POMCP
            B = sampleParticles(pm, b)
            #b = beliefParticles2Vector(pm, B)
            #println("b: ", float(map((x) -> x[2], b.belief)))

            a, Qv = selectAction(alg, pm, B)

            #b = beliefParticles2Vector(pm, BCBeliefParticles(alg.B[History()]))
            #println("b: ", float(map((x) -> x[2], b.belief)))
        else
            a, Qv = selectAction(alg, pm, b)
        end

        #println("b: ", float(map((x) -> x[2], b.belief)))
        println("p_hungry: ", p)
        println("Qv: ", float(map((x) -> round(x[2], 2), Qv)))
        println("action: ", a.action)
        println()
    end
end


function simulate_scenario(pm, alg)

    actions = [BCAction(:notfeed), BCAction(:feed), BCAction(:notfeed), BCAction(:notfeed), BCAction(:notfeed)]
    observations = [BCObservation(:crying), BCObservation(:notcrying), BCObservation(:notcrying), BCObservation(:notcrying), BCObservation(:crying)]

    b = BCBeliefVector([BCState(:nothungry) => 0.5, BCState(:hungry) => 0.5])

    if typeof(alg) == POMCP
        B = sampleParticles(pm, b)
        b = beliefParticles2Vector(pm, B)
    end

    println("b: ", float(map((x) -> x[2], b.belief)))

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

        println("b: ", float(map((x) -> x[2], b.belief)))

        if typeof(alg) == UCT || typeof(alg) == POMCP
            reinitialize(alg, a, o)
        end
    end

end


function simulate(pm, alg)

    s = BCState(:nothungry)
    b = BCBeliefVector([BCState(:nothungry) => 0.5, BCState(:hungry) => 0.5])

    if typeof(alg) == POMCP
        B = sampleParticles(pm, b)
        b = beliefParticles2Vector(pm, B)
    end

    R = 0

    println("time: 0, s: ", s.state, ", b: ", float(map((x) -> round(x[2], 2), b.belief)))

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

        if typeof(alg) == POMCP
            B = updateBelief(pm, BCBeliefParticles(getParticles(alg, a, o)))
            b = beliefParticles2Vector(pm, B)
        else
            b = updateBelief(pm, b, a, o)
        end

        println("time: ", i, ", s: ", s.state, ", Qv: ", float(map((x) -> round(x[2], 2), Qv)), ", a: ", a.action, ", r: ", r, ", o: ", o.observation, ", b: ", float(map((x) -> round(x[2], 2), b.belief)), ", R: ", R, ", s_: ", s_.state)

        s = s_

        if typeof(alg) == UCT || typeof(alg) == POMCP
            reinitialize(alg, a, o)
        end
    end
end


#srand(uint(time()))
srand(263)

pm = BabyCrying()

#alg = QMDP(pm, "babycrying_qmdp.pcy")
#alg = QMDP("babycrying_qmdp.pcy")
#alg = FIB(pm, "babycrying_fib.pcy")
#alg = FIB("babycrying_fib.pcy")

#alg = UCT(depth = 3, nloop_max = 10000, nloop_min = 10000, c = 20.)
alg = POMCP(depth = 3, nloop_max = 10000, nloop_min = 10000, c = 20.)

#test(pm, alg)
#validate(pm, alg)
#simulate_scenario(pm, alg)
simulate(pm, alg)


