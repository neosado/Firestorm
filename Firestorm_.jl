# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 12/02/2014

module Firestorm_

import Base.isequal, Base.hash, Base.copy
import Base.start, Base.done, Base.next

export Firestorm, FSState, FSAction, FSObservation, FSBelief, FSBeliefVector, FSBeliefParticles, History
export FSStateIter
export reward, observe, nextState, isEnd, isFeasible, sampleBelief, updateBelief
export tranProb, obsProb
export updateInternalStates


using POMDP_
using Wildfire_

using Base.Test
using Iterators


import POMDP_.reward
import POMDP_.observe
import POMDP_.nextState
import POMDP_.isEnd
import POMDP_.isFeasible
import POMDP_.sampleBelief
import POMDP_.updateBelief
import POMDP_.tranProb
import POMDP_.obsProb


immutable FSState <: State

    Position::(Int64, Int64)

    # a boolean variable indicating whether the cell is currently burning
    B::BurningMatrix

    # an integer variable indicating how much fuel is remaining in the cell
    #F::FuelLevelMatrix
end

type FSStateIter

    nrow::Int64
    ncol::Int64
    B_iter

    function FSStateIter(nrow, ncol)

        self = new()

        self.nrow = nrow
        self.ncol = ncol
        self.B_iter = product(repeated([false, true], nrow * ncol)...)

        return self
    end
end

function start(iter::FSStateIter)

    state = (1, 1, start(iter.B_iter))
end

function done(iter::FSStateIter, state)

    row, col, B_iter_state = state

    if col > iter.ncol
        return true
    end

    return false
end

function next(iter::FSStateIter, state)

    row, col, B_iter_state = state

    (B_array, B_iter_state) = next(iter.B_iter, B_iter_state)

    item = FSState((row, col), reshape([B_array...], (iter.nrow, iter.ncol)))

    if done(iter.B_iter, B_iter_state)
        B_iter_state = start(iter.B_iter)

        if row + 1 > iter.nrow
            row = 1
            col += 1
        else
            row += 1
        end
    end

    return item, (row, col, B_iter_state)
end

immutable FSAction <: Action
    action::Symbol
end

immutable FSObservation <: Observation
    observation::Symbol
end

abstract FSBelief <: Belief

immutable FSBeliefVector <: FSBelief
    belief::Dict{FSState, Float64}
end

immutable FSBeliefParticles <: FSBelief
    particles::Vector{FSState}
end


type Firestorm <: POMDP

    seed::Uint64

    nrow::Int64
    ncol::Int64

    states::FSStateIter
    nState::Int64

    actions::Vector{FSAction}
    nAction::Int64

    observations::Vector{FSObservation}
    nObservation::Int64

    reward_functype::Symbol


    wm::Wildfire

    uav_pos::(Int64, Int64)


    R::Array{Float64, 2}
    R_min::Float64
    R_max::Float64

    U::Array{Float64, 2}
    U_min::Float64
    U_max::Float64


    function Firestorm(n::Int64; seed::Int64 = 0)

        self = new()

        if seed != 0
            self.seed = uint(seed)
        else
            self.seed = uint(time())
        end

        srand(self.seed)

        nrow = n
        ncol = n

        self.nrow = nrow
        self.ncol = ncol

        self.states = FSStateIter(nrow, ncol)
        # uav_pos x B
        self.nState = (nrow * ncol) * 2^(nrow * ncol)
        # uav_pos x B x F
        #self.nState = (nrow * ncol) * 2^(nrow * ncol) * (F_max + 1)^(nrow * ncol)

        self.actions = [FSAction(:north), FSAction(:south), FSAction(:west), FSAction(:east)]
        self.nAction = 4

        self.observations = [FSObservation(:notburning), FSObservation(:burning)]
        self.nObservation = 2

        self.reward_functype = :type3

        self.wm = Wildfire(n, n, seed = seed)

        # UAV location
        side = rand(1:4)
        if side == 1
            self.uav_pos = (1, rand(1:ncol))
        elseif side == 2
            self.uav_pos = (nrow, rand(1:ncol))
        elseif side == 3
            self.uav_pos = (rand(1:nrow), 1)
        elseif side == 4
            self.uav_pos = (rand(1:nrow), ncol)
        end

        self.R = zeros(nrow, ncol)
        self.R_min = 0
        self.R_max = 30

        self.U = zeros(nrow, ncol)
        self.U_min = 0
        self.U_max = 30

        return self
    end
end


# P(s' | s, a)
function tranProb(fs::Firestorm, s::FSState, a::FSAction, s_::FSState)

    row, col = s.Position
    B = s.B

    row_, col_ = s_.Position
    B_ = s_.B


    drow = 0
    dcol = 0

    if a.action == :north && row != 1
        drow = -1
    elseif a.action == :south && row != fs.nrow
        drow = 1
    elseif a.action == :west && col != 1
        dcol = -1
    elseif a.action == :east && col != fs.ncol
        dcol = 1
    end

    if row + drow == row_ && col + dcol == col_
        prob = wfTranProb(fs.wm, B, B_)
    else
        prob = 0.
    end

    return prob
end


# P(o | s', a)
function obsProb(fs::Firestorm, s_::FSState, a::FSAction, o::FSObservation)

    row_, col_ = s_.Position
    B_ = s_.B


    if B_[row_, col_] == false && o.observation == :notburning
        prob = 1.
    elseif B_[row_, col_] == true && o.observation == :burning
        prob = 1.
    else
        prob = 0.
    end

    return prob
end


# R(s, a)
function reward(fs::Firestorm, s::FSState, a::FSAction, s_::FSState)

    row, col = s.Position
    B = s.B

    row_, col_ = s_.Position
    B_ = s_.B


    drow = 0
    dcol = 0

    if a.action == :north && row != 1
        drow = -1
    elseif a.action == :south && row != fs.nrow
        drow = 1
    elseif a.action == :west && col != 1
        dcol = -1
    elseif a.action == :east && col != fs.ncol
        dcol = 1
    end

    r = 0.

    if row + drow == row_ && col + dcol == col_
        # XXX assume that B_ is reachable from B.
        # QMDP, FIB, UCT, and POMCP satisfy the assumption.
        if !B[row_, col_] && B_[row_, col_]
            r = 10.
        #elseif B_[row_, col_]
        #    r = -10.
        #elseif !B_[row_, col_]
        #    r = -20.
        end
    end
        
    return r
end


# o ~ P(O | s', a)
function observe(fs::Firestorm, s_::FSState, a::FSAction)

    row_, col_ = s_.Position
    B_ = s_.B
    
    if B_[row_, col_]
        return FSObservation(:burning)
    else
        return FSObservation(:notburning)
    end
end


# s' ~ P(S | s, a)
function nextState(fs::Firestorm, s::FSState, a::FSAction)

    row, col = s.Position
    B = s.B


    drow = 0
    dcol = 0

    if a.action == :north
        @assert row != 1
        drow = -1
    elseif a.action == :south
        @assert row != fs.nrow
        drow = 1
    elseif a.action == :west
        @assert col != 1
        dcol = -1
    elseif a.action == :east
        @assert col != fs.ncol
        dcol = 1
    end

    B_ = wfNextState(fs.wm, B)

    return FSState((row + drow, col + dcol), B_)
end


function isEnd(fs::Firestorm, s::FSState)

    row, col = s.Position
    B = s.B

    bBurning = B[1, 1]
    bEnd = true

    for j = 1:fs.ncol
        for i = 1:fs.nrow
            if B[i, j] != bBurning
                bEnd = false

                break
            end
        end

        if !bEnd
            break
        end
    end

    return bEnd
end


function isFeasible(fs::Firestorm, s::FSState, a::FSAction)

    row, col = s.Position

    if a.action == :north && row == 1
        return false
    elseif a.action == :south && row == fs.nrow
        return false
    elseif a.action == :west && col == 1
        return false
    elseif a.action == :east && col == fs.ncol
        return false
    end

    return true
end


# s ~ b
function sampleBelief(fs::Firestorm, b::FSBeliefVector)

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

function sampleBelief(fs::Firestorm, b::FSBeliefParticles)

    s = b.particles[rand(1:length(b.particles))]

    return copy(s)
end


# b' = B(b, a, o)
function updateBelief(fs::Firestorm, b::FSBeliefVector, a::FSAction, o::FSObservation)

    # b'(s') = O(o | s', a) \sum_s T(s' | s, a) b(s)

    belief_ = Dict{FSState, Float64}()

    sum_belief = 0.
    for s_ in keys(b.belief)
        sum_ = 0.

        for (s, v) in b.belief
            sum_ += tranProb(fs, s, a, s_) * v
        end

        belief_[s_] = obsProb(fs, s_, a, o) * sum_
        sum_belief += belief_[s_]
    end

    for s_ in keys(belief_)
        belief_[s_] /= sum_belief
    end

    @test length(belief_) == fs.nState
    sum_ = 0.
    for v in values(belief_)
        sum_ += v
    end
    @test_approx_eq sum_ 1.

    return FSBeliefVector(belief_)
end

function updateBelief(fs::Firestorm, b::FSBeliefParticles)

    return b
end


function updateInternalStates(fs::Firestorm, s::FSState, a::FSAction, s_::FSState)

    fs.uav_pos = s_.Position
end


function isequal(s1::FSState, s2::FSState)

    return isequal(s1.Position, s2.Position) && isequal(s1.B, s2.B)
end

function ==(s1::FSState, s2::FSState)

    return (s1.Position == s2.Position) && (s1.B == s2.B)
end

function hash(s::FSState, h::Uint64 = zero(Uint64))

    return hash(s.B, hash(s.Position, h))
end

function copy(s::FSState)

    return FSState(s.Position, copy(s.B))
end


function isequal(a1::FSAction, a2::FSAction)

    return isequal(a1.action, a2.action)
end

function ==(a1::FSAction, a2::FSAction)

    return (a1.action == a2.action)
end

function hash(a::FSAction, h::Uint64 = zero(Uint64))

    return hash(a.action, h)
end


function isequal(o1::FSObservation, o2::FSObservation)

    return isequal(o1.observation, o2.observation)
end

function ==(o1::FSObservation, o2::FSObservation)

    return (o1.observation == o2.observation)
end

function hash(o::FSObservation, h::Uint64 = zero(Uint64))

    return hash(o.observation, h)
end


function isequal(k1::(History, FSAction), k2::(History, FSAction))

    return isequal(k1[1], k2[1]) && isequal(k1[2], k2[2])
end

function ==(k1::(History, FSAction), k2::(History, FSAction))

    return (k1[1] == k2[1]) && (k1[2] == k2[2])
end

function hash(k::(History, FSAction), h::Uint64 = zero(Uint64))

    return hash(k[2], hash(k[1], h))
end


end


