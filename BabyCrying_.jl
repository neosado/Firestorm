# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

# Mykel J. Kochenderfer, Decision Making Under Uncertainty: Theory and Application, MIT Press, 2014.

module BabyCrying_

import Base.isequal, Base.hash

export BabyCrying, BCState, BCAction, BCObservation, BCBelief, BCBeliefVector, BCBeliefParticles, History
export reward, observe, nextState, isEnd, sampleBelief, updateBelief
export tranProb, obsProb


using POMDP_
using Base.Test


import POMDP_.reward
import POMDP_.observe
import POMDP_.nextState
import POMDP_.isEnd
import POMDP_.sampleBelief
import POMDP_.updateBelief
import POMDP_.tranProb
import POMDP_.obsProb


immutable BCState <: State
    state::Symbol
end

immutable BCAction <: Action
    action::Symbol
end

immutable BCObservation <: Observation
    observation::Symbol
end

abstract BCBelief <: Belief

immutable BCBeliefVector <: BCBelief
    belief::Dict{BCState, Float64}
end

immutable BCBeliefParticles <: BCBelief
    particles::Vector{BCState}
end


# decide when to feed the baby on the basis of whether the baby is crying
type BabyCrying <: POMDP

    states::Vector{BCState}
    nState::Int64

    actions::Vector{BCAction}
    nAction::Int64

    observations::Vector{BCObservation}
    nObservation::Int64

    reward_functype::Symbol


    function BabyCrying()

        self = new()

        self.states = [BCState(:nothungry), BCState(:hungry)]
        self.nState = 2

        self.actions = [BCAction(:notfeed), BCAction(:feed)]
        self.nAction = 2

        self.observations = [BCObservation(:notcrying), BCObservation(:crying)]
        self.nObservation = 2

        self.reward_functype = :type2

        srand(uint(time()))

        return self
    end
end


# P(s' | s, a)
function tranProb(bc::BabyCrying, s::BCState, a::BCAction, s_::BCState)

    # if feed the baby, the baby stops being hungry at the next time step
    # if not hungry and not feed, 10% chance that the baby may become hungry at the next time step
    # once hungry, the baby continues being hungry until fed

    # s a s_ prob
    # 0 0 0  0.9
    # 0 0 1  0.1
    # 0 1 0  1.
    # 0 1 1  0.
    # 1 0 0  0.
    # 1 0 1  1.
    # 1 1 0  1.
    # 1 1 1  0.

    if s.state == :nothungry && a.action == :notfeed
        if s_.state == :nothungry
            prob = 0.9
        elseif s_.state == :hungry
            prob = 0.1
        end
    elseif s.state == :nothungry && a.action == :feed
        if s_.state == :nothungry
            prob = 1.
        elseif s_.state == :hungry
            prob = 0.
        end
    elseif s.state == :hungry && a.action == :notfeed
        if s_.state == :nothungry
            prob = 0.
        elseif s_.state == :hungry
            prob = 1.
        end
    elseif s.state == :hungry && a.action == :feed
        if s_.state == :nothungry
            prob = 1.
        elseif s_.state == :hungry
            prob = 0.
        end
    end

    return prob
end


# P(o | s', a)
function obsProb(bc::BabyCrying, s_::BCState, a::BCAction, o::BCObservation)

    # 10% chance that the baby cries when not hungry
    # 80% chance that the baby cries when hungry

    # o s' a prob
    # 0 0  0 0.9
    # 1 0  0 0.1
    # 0 1  0 0.2
    # 1 1  0 0.8

    if a.action == :notfeed || (a.action == :feed && s_.state == :nothungry)
        if s_.state == :nothungry
            if o.observation == :notcrying
                prob = 0.9
            elseif o.observation == :crying
                prob = 0.1
            end
        elseif s_.state == :hungry
            if o.observation == :notcrying
                prob = 0.2
            elseif o.observation == :crying
                prob = 0.8
            end
        end
    else
        prob = 0.
    end

    return prob
end


# R(s, a)
function reward(bc::BabyCrying, s::BCState, a::BCAction)

    # hungry: -10
    # feed: -5
    # hungry and feed: -15

    r = 0

    if s.state == :hungry && a.action == :feed
        r = -15
    elseif s.state == :hungry
        r = -10
    elseif a.action == :feed
        r = -5
    end
        
    return r
end


# o ~ P(O | s', a)
function observe(bc::BabyCrying, s_::BCState, a::BCAction)

    rv = rand()
    p_cs = 0.

    for o in bc.observations
        p_cs += obsProb(bc, s_, a, o)

        if rv < p_cs
            return o
        end
    end

    @assert false
end


# s' ~ P(S | s, a)
function nextState(bc::BabyCrying, s::BCState, a::BCAction)

    rv = rand()
    p_cs = 0.

    for i = 1:bc.nState
        s_ = bc.states[i]
        p_cs += tranProb(bc, s, a, s_)

        if rv < p_cs
            return s_
        end
    end

    @assert false
end


function isEnd(bc::BabyCrying, s::BCState)

    return false
end


# s ~ b
function sampleBelief(bc::BabyCrying, b::BCBeliefVector)

    rv = rand()

    sum_ = 0.
    for (s, v) in b.belief
        sum_ += v

        if rv < sum_
            return s
        end
    end

    @assert false
end

function sampleBelief(bc::BabyCrying, b::BCBeliefParticles)

    s = b.particles[rand(1:length(b.particles))]

    return s
end

# b' = B(b, a, o)
function updateBelief(bc::BabyCrying, b::BCBeliefVector, a::BCAction, o::BCObservation)

    # b'(s') = O(o | s', a) \sum_s T(s' | s, a) b(s)

    belief_ = Dict{BCState, Float64}()

    sum_belief = 0.
    for s_ in keys(b.belief)
        sum_ = 0.

        for (s, v) in b.belief
            sum_ += tranProb(bc, s, a, s_) * v
        end

        belief_[s_] = obsProb(bc, s_, a, o) * sum_
        sum_belief += belief_[s_]
    end

    for s_ in keys(belief_)
        belief_[s_] /= sum_belief
    end

    @test length(belief_) == bc.nState
    sum_ = 0.
    for v in values(belief_)
        sum_ += v
    end
    @test_approx_eq sum_ 1.

    return BCBeliefVector(belief_)
end

function updateBelief(bc::BabyCrying, b::BCBeliefParticles)

    return b
end




function isequal(s1::BCState, s2::BCState)

    return isequal(s1.state, s2.state)
end

function ==(s1::BCState, s2::BCState)

    return (s1.state == s2.state)
end

function hash(s::BCState, h::Uint64 = zero(Uint64))

    return hash(s.state, h)
end


function isequal(a1::BCAction, a2::BCAction)

    return isequal(a1.action, a2.action)
end

function ==(a1::BCAction, a2::BCAction)

    return (a1.action == a2.action)
end

function hash(a::BCAction, h::Uint64 = zero(Uint64))

    return hash(a.action, h)
end


function isequal(o1::BCObservation, o2::BCObservation)

    return isequal(o1.observation, o2.observation)
end

function ==(o1::BCObservation, o2::BCObservation)

    return (o1.observation == o2.observation)
end

function hash(o::BCObservation, h::Uint64 = zero(Uint64))

    return hash(o.observation, h)
end


end


