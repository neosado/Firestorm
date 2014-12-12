# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

# Mykel J. Kochenderfer, Decision Making Under Uncertainty: Theory and Application, MIT Press, 2014.

module CryingBaby_

import Base.isequal, Base.hash

export CryingBaby, CBState, CBAction, CBObservation, CBBelief, CBBeliefVector, CBBeliefParticles, History
export reward, observe, nextState, isEnd, isFeasible, sampleBelief, updateBelief
export tranProb, obsProb


using POMDP_
using Base.Test


import POMDP_.reward
import POMDP_.observe
import POMDP_.nextState
import POMDP_.isEnd
import POMDP_.isFeasible
import POMDP_.sampleBelief
import POMDP_.updateBelief
import POMDP_.tranProb
import POMDP_.obsProb


immutable CBState <: State
    state::Symbol
end

immutable CBAction <: Action
    action::Symbol
end

immutable CBObservation <: Observation
    observation::Symbol
end

abstract CBBelief <: Belief

immutable CBBeliefVector <: CBBelief
    belief::Dict{CBState, Float64}
end

immutable CBBeliefParticles <: CBBelief
    particles::Vector{CBState}
end


# decide when to feed the baby on the basis of whether the baby is crying
type CryingBaby <: POMDP

    states::Vector{CBState}
    nState::Int64

    actions::Vector{CBAction}
    nAction::Int64

    observations::Vector{CBObservation}
    nObservation::Int64

    reward_functype::Symbol


    function CryingBaby()

        self = new()

        self.states = [CBState(:nothungry), CBState(:hungry)]
        self.nState = 2

        self.actions = [CBAction(:notfeed), CBAction(:feed)]
        self.nAction = 2

        self.observations = [CBObservation(:notcrying), CBObservation(:crying)]
        self.nObservation = 2

        self.reward_functype = :type2

        srand(uint(time()))

        return self
    end
end


# P(s' | s, a)
function tranProb(cb::CryingBaby, s::CBState, a::CBAction, s_::CBState)

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
function obsProb(cb::CryingBaby, s_::CBState, a::CBAction, o::CBObservation)

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
function reward(cb::CryingBaby, s::CBState, a::CBAction)

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
function observe(cb::CryingBaby, s_::CBState, a::CBAction)

    rv = rand()
    p_cs = 0.

    for o in cb.observations
        p_cs += obsProb(cb, s_, a, o)

        if rv < p_cs
            return o
        end
    end

    @assert false
end


# s' ~ P(S | s, a)
function nextState(cb::CryingBaby, s::CBState, a::CBAction)

    rv = rand()
    p_cs = 0.

    for i = 1:cb.nState
        s_ = cb.states[i]
        p_cs += tranProb(cb, s, a, s_)

        if rv < p_cs
            return s_
        end
    end

    @assert false
end


function isEnd(cb::CryingBaby, s::CBState)

    return false
end


function isFeasible(cb::CryingBaby, s::CBState, a::CBAction)

    return true
end


# s ~ b
function sampleBelief(cb::CryingBaby, b::CBBeliefVector)

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

function sampleBelief(cb::CryingBaby, b::CBBeliefParticles)

    s = b.particles[rand(1:length(b.particles))]

    return s
end


# b' = B(b, a, o)
function updateBelief(cb::CryingBaby, b::CBBeliefVector, a::CBAction, o::CBObservation)

    # b'(s') = O(o | s', a) \sum_s T(s' | s, a) b(s)

    belief_ = Dict{CBState, Float64}()

    sum_belief = 0.
    for s_ in keys(b.belief)
        sum_ = 0.

        for (s, v) in b.belief
            sum_ += tranProb(cb, s, a, s_) * v
        end

        belief_[s_] = obsProb(cb, s_, a, o) * sum_
        sum_belief += belief_[s_]
    end

    for s_ in keys(belief_)
        belief_[s_] /= sum_belief
    end

    @test length(belief_) == cb.nState
    sum_ = 0.
    for v in values(belief_)
        sum_ += v
    end
    @test_approx_eq sum_ 1.

    return CBBeliefVector(belief_)
end

function updateBelief(cb::CryingBaby, b::CBBeliefParticles)

    return b
end




function isequal(s1::CBState, s2::CBState)

    return isequal(s1.state, s2.state)
end

function ==(s1::CBState, s2::CBState)

    return (s1.state == s2.state)
end

function hash(s::CBState, h::Uint64 = zero(Uint64))

    return hash(s.state, h)
end


function isequal(a1::CBAction, a2::CBAction)

    return isequal(a1.action, a2.action)
end

function ==(a1::CBAction, a2::CBAction)

    return (a1.action == a2.action)
end

function hash(a::CBAction, h::Uint64 = zero(Uint64))

    return hash(a.action, h)
end


function isequal(o1::CBObservation, o2::CBObservation)

    return isequal(o1.observation, o2.observation)
end

function ==(o1::CBObservation, o2::CBObservation)

    return (o1.observation == o2.observation)
end

function hash(o::CBObservation, h::Uint64 = zero(Uint64))

    return hash(o.observation, h)
end


end


