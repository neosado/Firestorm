# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

# Partially Observable Markov Decision Process
module POMDP_

import Base.isequal, Base.hash

export POMDP, State, Action, Observation, Belief, History
export reward, observe, nextState, isEnd, sampleBelief, updateBelief


abstract POMDP

reward(pm::POMDP) = error("$(typeof(pm)) does not implement reward()")
observe(pm::POMDP) = error("$(typeof(pm)) does not implement observe()")
nextState(pm::POMDP) = error("$(typeof(pm)) does not implement nextState()")
isEnd(pm::POMDP) = error("$(typeof(pm)) does not implement isEnd()")
sampleBelief(pm::POMDP) = error("$(typeof(pm)) does not implement sampleBelief()")
updateBelief(pm::POMDP) = error("$(typeof(pm)) does not implement updateBelief()")


abstract State
abstract Action
abstract Observation
abstract Belief

type History

    history::Vector{Any}


    function History(history = {})

        self = new()
        self.history = history

        return self
    end
end

function isequal(h1::History, h2::History)

    if length(h1.history) != length(h2.history)
        return false
    else
        return all([isequal(e1, e2) for (e1, e2) in zip(h1.history, h2.history)])
    end
end

function ==(h1::History, h2::History)

    if length(h1.history) != length(h2.history)
        return false
    else
        return all([e1 == e2 for (e1, e2) in zip(h1.history, h2.history)])
    end
end

function hash(hist::History, h::Uint64 = zero(Uint64))

    h = hash(nothing, h)

    for h_ in hist.history
        h = hash(h_, h)
    end

    return h
end


end


