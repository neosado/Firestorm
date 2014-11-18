# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

# T. Smith and R.G. Simmons, "Heuristic Search Value Iteration for POMDPs," in Conference on Uncertainty in Artificial Intelligence (UAI), 2004.

module RockSample_

import Base.isequal, Base.hash, Base.copy

export RockSample, RSState, RSAction, RSObservation, RSBelief, RSBeliefVector, RSBeliefParticles, History
export reward, observe, nextState, isEnd, sampleBelief, updateBelief
export updateInternalStates, rock2ind


using POMDP_
using Base.Test


import POMDP_.reward
import POMDP_.observe
import POMDP_.nextState
import POMDP_.isEnd
import POMDP_.sampleBelief
import POMDP_.updateBelief


immutable RSState <: State

    Position::(Int64, Int64)
    RockTypes::Vector{Symbol}
end

immutable RSAction <: Action
    action::Symbol
end

immutable RSObservation <: Observation
    observation::Symbol
end

abstract RSBelief <: Belief

immutable RSBeliefVector <: RSBelief
    belief::Dict{RSState, Float64}
end

immutable RSBeliefParticles <: RSBelief
    particles::Vector{RSState}
end


type RockSample <: POMDP

    seed::Uint64

    n::Int64
    k::Int64

    states::Vector{RSState}
    nState::Int64

    actions::Vector{RSAction}
    nAction::Int64

    p_actions::Vector{RSAction}
    npAction::Int64

    obs::Vector{RSObservation}
    nObservation::Int64

    rover_pos::(Int64, Int64)

    rock_types::Dict{Symbol, Symbol}            # rock -> type
    rock_locs::Dict{Symbol, (Int64, Int64)}     # rock -> loc
    rock_sampled::Dict{Symbol, Bool}            # rock -> bool
    loc2rock::Dict{(Int64, Int64), Symbol}      # loc -> rock


    function RockSample(n::Int64, k::Int64; seed::Int64 = 0)

        if seed != 0
            srand(uint(seed))
        else
            srand(uint(time()))
        end

        self = new()

        self.seed = uint(seed)

        self.n = n
        self.k = k

        self.states = []
        self.nState = n * n * 2 ^ k

        self.actions = [RSAction(:north), RSAction(:south), RSAction(:east), RSAction(:west), RSAction(:sample)]
        for i = 1:k
            push!(self.actions, RSAction(symbol("check$i")))
        end
        self.nAction = 5 + k

        self.p_actions = [RSAction(:north), RSAction(:south), RSAction(:east), RSAction(:west), RSAction(:sample)]
        for i = 1:k
            push!(self.p_actions, RSAction(symbol("check$i")))
        end
        self.npAction = 5 + k

        self.obs = [RSObservation(:none), RSObservation(:good), RSObservation(:bad)]
        self.nObservation = 3

        # XXX Debug
        #self.rover_pos = (rand(1:n), rand(1:n))
        self.rover_pos = (rand(1:n), rand(1:2))

        self.rock_types = Dict{Symbol, Symbol}()
        self.rock_locs = Dict{Symbol, (Int64, Int64)}()
        self.rock_sampled = Dict{Symbol, Bool}()
        self.loc2rock = Dict{(Int64, Int64), Symbol}()

        i = 0
        while i < k
            row = rand(1:n)
            col = rand(1:n)

            types = [:good, :bad]
            type_ = types[rand(1:length(types))]

            if !haskey(self.loc2rock, (row, col))
                i += 1
                self.rock_types[symbol("rock$i")] = type_
                self.rock_locs[symbol("rock$i")] = (row, col)
                self.rock_sampled[symbol("rock$i")] = false
                self.loc2rock[(row, col)] = symbol("rock$i")
            end
        end

        return self
    end
end


isCheckAction(a::RSAction) = (length(string(a.action)) == 6 && string(a.action)[1:5] == "check")

act2rock(a::RSAction) = symbol("rock" * string(string(a.action)[6]))
act2rockIndex(a::RSAction) = int(string(string(a.action)[6]))
rock2ind(rock::Symbol) = int(string(string(rock)[5]))


# P(s' | s, a)
function prob_tran(rs::RockSample, s::RSState, a::RSAction, s_::RSState)

    row, col = s.Position
    rock_types = s.RockTypes

    row_, col_ = s_.Position
    rock_types_ = s_.RockTypes

    prob = 0.

    if a.action == :north
        if row != 1 && row - 1 == row_ && col == col_ && rock_types == rock_types_
            prob = 1.
        elseif row == 1 && row == row_ && col == col_ && rock_types == rock_types_
            prob = 1.
        end
    elseif a.action == :south
        if row != rs.n && row + 1 == row_ && col == col_ && rock_types == rock_types_
            prob = 1.
        elseif row == rs.n && row == row_ && col == col_ && rock_types == rock_types_
            prob = 1.
        end
    elseif a.action == :east
        if row == row_ && col + 1 == col_ && rock_types == rock_types_
            prob = 1.
        end
    elseif a.action == :west
        if col != 1 && row == row_ && col - 1 == col_ && rock_types == rock_types_
            prob = 1.
        elseif col == 1 && row == row_ && col == col_ && rock_types == rock_types_
            prob = 1.
        end
    elseif a.action == :sample
        if row == row_ && col == col_
            if haskey(rs.loc2rock, (row, col))
                rock = rs.loc2rock[(row, col)]
                rock_index = rock2ind(rock)

                rock_types = copy(s.RockTypes)
                rock_types[rock_index] = :bad

                if rock_types == rock_types_
                    prob = 1.
                end
            elseif rock_types == rock_types_
                prob = 1.
            end
        end
    else
        if row == row_ && col == col_ && rock_types == rock_types_
            prob = 1.
        end
    end

    return prob
end


# P(o | s', a)
function prob_obs(rs::RockSample, s_::RSState, a::RSAction, o::RSObservation)

    pos_ = s_.Position
    rock_types_ = s_.RockTypes

    prob = 0.

    if isCheckAction(a)
        rock = act2rock(a)
        rock_index = rock2ind(rock)

        rock_loc = rs.rock_locs[rock]
        # XXX Debug
        #rock_type = rock_types_[rock_index]
        rock_type = rs.rock_types[rock]

        dist = sqrt((pos_[1] - rock_loc[1])^2 + (pos_[2] - rock_loc[2])^2)

        eta = exp(-dist)

        p_correct = 0.5 + 0.5 * eta

        if rock_type == :good
            if o.observation == :good
                prob = p_correct
            elseif o.observation == :bad
                prob = 1. - p_correct
            end
        elseif rock_type == :bad
            if o.observation == :good
                prob = 1. - p_correct
            elseif o.observation == :bad
                prob = p_correct
            end
        end
    elseif o.observation == :none
        prob = 1.
    end

    return prob
end


# R(s, a)
function reward(rs::RockSample, s::RSState, a::RSAction)

    row, col = s.Position
    rock_types = s.RockTypes

    r = 0.

    if a.action == :north && row == 1
        r = -100.
    elseif a.action == :south && row == rs.n
        r = -100.
    elseif a.action == :east && col == rs.n
        r = 10.
    elseif a.action == :west && col == 1
        r = -100.
    elseif a.action == :sample
        if haskey(rs.loc2rock, (row, col))
            rock = rs.loc2rock[(row, col)]
            # XXX Debug
            #rock_index = rock2ind(rock)
            #rock_type = rock_types[rock_index]
            rock_type = rs.rock_types[rock]

            if rock_type == :good
                r = 10.
            elseif rock_type == :bad
                r = -10.
            end
        end
    end
        
    return r
end


# o ~ P(O | s', a)
function observe(rs::RockSample, s_::RSState, a::RSAction)

    rv = rand()
    p_cs = 0.

    for o in rs.obs
        p_cs += prob_obs(rs, s_, a, o)

        if rv < p_cs
            return o
        end
    end

    @assert false
end


# s' ~ P(S | s, a)
function nextState(rs::RockSample, s::RSState, a::RSAction)

    row, col = s.Position
    rock_types = copy(s.RockTypes)

    if a.action == :north
        if row != 1
            row -= 1
        end
    elseif a.action == :south
        if row != rs.n
            row += 1
        end
    elseif a.action == :east
        col += 1
    elseif a.action == :west
        if col != 1
            col -= 1
        end
    elseif a.action == :sample
        if haskey(rs.loc2rock, (row, col))
            rock = rs.loc2rock[(row, col)]
            rock_types[rock2ind(rock)] = :bad
        end
    end

    s_ = RSState((row, col), rock_types)

    return s_
end


function isEnd(rs::RockSample, s::RSState)

    row, col = s.Position

    if col == rs.n + 1
        return true
    else
        return false
    end
end


# s ~ b
function sampleBelief(rs::RockSample, b::RSBeliefVector)

    rv = rand()

    sum_ = 0.
    for (s, v) in b.belief
        sum_ += v

        if rv < sum_
            return copy(s)
        end
    end

    @assert false
end

function sampleBelief(rs::RockSample, b::RSBeliefParticles)

    s = b.particles[rand(1:length(b.particles))]

    return s
end


# b' = B(b, a, o)
function updateBelief(rs::RockSample, b::RSBeliefVector, a::RSAction, o::RSObservation)

    # b'(s') = O(o | s', a) \sum_s T(s' | s, a) b(s)

    belief_ = Dict{RSState, Float64}()

    sum_belief = 0.
    for s_ in keys(b.belief)
        sum_ = 0.

        for (s, v) in b.belief
            sum_ += prob_tran(rs, s, a, s_) * v
        end

        belief_[s_] = prob_obs(rs, s_, a, o) * sum_
        sum_belief += belief_[s_]
    end

    for s_ in keys(belief_)
        belief_[s_] /= sum_belief
    end

    @test length(belief_) == rs.nState
    sum_ = 0.
    for v = values(belief_)
        sum_ += v
    end
    @test_approx_eq sum_ 1.

    return RSBeliefVector(belief_)
end

function updateBelief(rs::RockSample, b::RSBeliefParticles)

    return b
end


function updateInternalStates(rs::RockSample, s::RSState, a::RSAction, s_::RSState)

    rs.rover_pos = s_.Position

    if a.action == :sample && haskey(rs.loc2rock, s.Position)
        rock = rs.loc2rock[s.Position]
        rs.rock_types[rock] = :bad
        rs.rock_sampled[rock] = true
    end
end


function isequal(s1::RSState, s2::RSState)

    return isequal(s1.Position, s2.Position) && isequal(s1.RockTypes, s2.RockTypes)
end

function ==(s1::RSState, s2::RSState)

    return (s1.Position == s2.Position) && (s1.RockTypes == s2.RockTypes)
end

function hash(s::RSState, h::Uint64 = zero(Uint64))

    return hash(s.RockTypes, hash(s.Position, h))
end

function copy(s::RSState)

    return RSState(s.Position, copy(s.RockTypes))
end


function isequal(a1::RSAction, a2::RSAction)

    return isequal(a1.action, a2.action)
end

function ==(a1::RSAction, a2::RSAction)

    return (a1.action == a2.action)
end

function hash(a::RSAction, h::Uint64 = zero(Uint64))

    return hash(a.action, h)
end


function isequal(o1::RSObservation, o2::RSObservation)

    return isequal(o1.observation, o2.observation)
end

function ==(o1::RSObservation, o2::RSObservation)

    return (o1.observation == o2.observation)
end

function hash(o::RSObservation, h::Uint64 = zero(Uint64))

    return hash(o.observation, h)
end


function isequal(k1::(History, RSAction), k2::(History, RSAction))

    return isequal(k1[1], k2[1]) && isequal(k1[2], k2[2])
end

function ==(k1::(History, RSAction), k2::(History, RSAction))

    return (k1[1] == k2[1]) && (k1[2] == k2[2])
end

function hash(k::(History, RSAction), h::Uint64 = zero(Uint64))

    return hash(k[2], hash(k[1], h))
end


end


