# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/13/2015

module ScenarioOne_

export ScenarioOne, ScenarioOneState
export initTrajectories, updateLocations, updateStatus, getReward, isEndState


using Wildfire_

using Base.Test


type ScenarioOne

    seed::Uint64

    nrow::Int64
    ncol::Int64

    p_uav_landing_crash::Float64

    wm::Wildfire


    uav_loc::(Int64, Int64)
    uav_policy::Symbol

    uav_velocity::Float64

    uav_base_loc::(Int64, Int64)

    uav_path::Array{Float64, 2}
    uav_dpath::Vector{(Int64, Int64)}


    aircraft_start_loc::(Int64, Int64)
    aircraft_end_loc::(Int64, Int64)

    aircraft_velocity::Float64

    aircraft_path::Array{Float64, 2}
    aircraft_dpath::Vector{(Int64, Int64)}


    function ScenarioOne(n::Int64; seed::Int64 = 0, p_uav_landing_crash::Float64 = 0.01, p_fire::Float64 = 0.06, uav_policy::Symbol = :back, uav_velocity::Float64 = 1., aircraft_velocity::Float64 = 1.)

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

        self.wm = Wildfire(n, n, seed = seed, init_loc = (int(nrow / 2), int(ncol / 2)), p_fire = p_fire)

        simulate_wildfire(self.wm, 10)

        #self.uav_loc = (rand(1:nrow), rand(1:ncol))

        locs = getUAVInitLocs(self.wm)
        self.uav_loc = locs[rand(1:length(locs))]

        self.uav_policy = uav_policy

        self.uav_velocity = uav_velocity

        Lot = [(1, j) for j = 1:ncol]
        append!(Lot, [(nrow, j) for j = 1:ncol])
        append!(Lot, [(i, 1) for i = 2:nrow-1])
        append!(Lot, [(i, ncol) for i = 2:nrow-1])
        nLot = length(Lot)

        self.uav_base_loc = Lot[rand(1:nLot)]

        #self.aircraft_start_loc = Lot[rand(1:nLot)]
        #self.aircraft_end_loc = Lot[rand(1:nLot)]
        #@test self.aircraft_start_loc != self.aircraft_end_loc

        side = rand(1:4)
        if side == 1
            self.aircraft_start_loc = (1, rand(1:ncol))
            self.aircraft_end_loc = (nrow, ncol - self.aircraft_start_loc[2])
        elseif side == 2
            self.aircraft_start_loc = (nrow, rand(1:ncol))
            self.aircraft_end_loc = (1, ncol - self.aircraft_start_loc[2])
        elseif side == 3
            self.aircraft_start_loc = (rand(1:nrow), 1)
            self.aircraft_end_loc = (nrow - self.aircraft_start_loc[1], ncol)
        elseif side == 4
            self.aircraft_start_loc = (rand(1:nrow), ncol)
            self.aircraft_end_loc = (nrow - self.aircraft_start_loc[1], 1)
        end

        self.aircraft_velocity = aircraft_velocity

        return self
    end
end


type ScenarioOneState

    uav_loc::(Int64, Int64)
    uav_status::Symbol

    aircraft_loc::(Int64, Int64)
    aircraft_status::Symbol


    function ScenarioOneState(s1::ScenarioOne)

        self = new()

        self.uav_loc = s1.uav_loc
        self.uav_status = :flying

        self.aircraft_loc = s1.aircraft_start_loc
        self.aircraft_status = :flying

        return self
    end
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


function getUAVInitLocsHelper(M::Array{Bool, 2})

    nrow, ncol = size(M)

    L = zeros(Bool, nrow, ncol)

    for i = 2:nrow-1
        for j = 2:ncol-1
            if M[i, j] || M[i + 1, j] || M[i - 1, j] || M[i, j + 1] || M[i, j - 1]
                L[i, j] = true
            end
        end
    end

    return L
end

function getUAVInitLocs(wm::Wildfire)

    L = deepcopy(wm.B)
    L = getUAVInitLocsHelper(L)
    L = getUAVInitLocsHelper(L)

    nrow, ncol = size(L)
    locs = (Int64, Int64)[]

    for i = 1:nrow
        for j = 1:wm.ncol
            if L[i, j]
                push!(locs, (i, j))
            end
        end
    end

    return locs
end


function generate_aircraft_trajectory(s1::ScenarioOne)

    P = [s1.aircraft_start_loc, s1.aircraft_end_loc]

    n = length(P)
    d = n - 1
    nc = s1.nrow * s1.ncol

    path = zeros(nc, 2)

    k = 1
    for t = linspace(0, 1, nc)
        # Bezier curve using Bernstein basis polynomials
        # http://en.wikipedia.org/wiki/B챕zier_curve
        #path[k, 1] = sum([factorial(d) / (factorial(i) * factorial(d-i)) * t^i * (1-t)^(d-i) * P[i+1][1] for i = 0:d])
        #path[k, 2] = sum([factorial(d) / (factorial(i) * factorial(d-i)) * t^i * (1-t)^(d-i) * P[i+1][2] for i = 0:d])

        # straight line
        path[k, 1] = t * s1.aircraft_end_loc[1] + (1 - t) * s1.aircraft_start_loc[1]
        path[k, 2] = t * s1.aircraft_end_loc[2] + (1 - t) * s1.aircraft_start_loc[2]

        k += 1
    end

    # debug
    #println(P)
    #println(path')
    #plot(path[:,1], path[:,2])
    #readline()

    return path
end


function generate_uav_trajectory(s1::ScenarioOne)

    if s1.uav_policy == :back
        P = [s1.uav_loc, s1.uav_base_loc]

        nc = s1.nrow * s1.ncol

        path = zeros(nc, 2)

        k = 1
        for t = linspace(0, 1, nc)
            path[k, 1] = t * s1.uav_base_loc[1] + (1 - t) * s1.uav_loc[1]
            path[k, 2] = t * s1.uav_base_loc[2] + (1 - t) * s1.uav_loc[2]

            k += 1
        end
    else
        path = nothing
    end

    # debug
    #println(P)
    #println(path')
    #plot(path[:,1], path[:,2])
    #readline()

    return path
end


function discretize_path(path::Array{Float64, 2}, velocity)

    dpath = (Int64, Int64)[]

    prev_loc = path[1, :]
    push!(dpath, (int(prev_loc[1]), int(prev_loc[2])))

    t = 1
    dist = 0.

    for i = 2:size(path, 1)
        curr_loc = path[i, :]

        dist += norm(curr_loc - prev_loc)

        if dist >= velocity
            push!(dpath, (int(curr_loc[1]), int(curr_loc[2])))

            dist = 0.
            t += 1
        end

        prev_loc = curr_loc
    end

    if dpath[end] != (path[end, 1], path[end, 2])
        push!(dpath, (int(path[end, 1]), int(path[end, 2])))
    end

    return dpath
end


function initTrajectories(s1::ScenarioOne)

    s1.aircraft_path = generate_aircraft_trajectory(s1)
    s1.aircraft_dpath = discretize_path(s1.aircraft_path, s1.aircraft_velocity)

    s1.uav_path = generate_uav_trajectory(s1)
    s1.uav_dpath = discretize_path(s1.uav_path, s1.uav_velocity)
end

function getUAVLoc(s1::ScenarioOne, t::Int64)

    if t + 1 > length(s1.uav_dpath)
        return nothing
    else
        return s1.uav_dpath[t + 1]
    end
end


function getAircraftLoc(s1::ScenarioOne, t::Int64)

    if t + 1 > length(s1.aircraft_dpath)
        return nothing
    else
        return s1.aircraft_dpath[t + 1]
    end
end


function updateLocations(s1::ScenarioOne, s1state::ScenarioOneState, t::Int64)

    s1state.uav_loc = getUAVLoc(s1, t)
    s1state.aircraft_loc = getAircraftLoc(s1, t)
end


function updateStatus(s1::ScenarioOne, s1state::ScenarioOneState)

    if s1state.uav_loc == s1.uav_base_loc
        if rand() < s1.p_uav_landing_crash
            s1state.uav_status = :crashed
        else
            s1state.uav_status = :landed
        end
    end

    if s1state.aircraft_loc == s1.aircraft_end_loc
        s1state.aircraft_status = :left
    end
end


function getReward(s1::ScenarioOne, s1state::ScenarioOneState)

    if s1state.uav_status == :crashed
        return -10
    end

    dist = norm([s1state.uav_loc[1] - s1state.aircraft_loc[1], s1state.uav_loc[2] - s1state.aircraft_loc[2]])

    if dist < 1.
        return -100
    elseif dist < 2.
        return -20
    elseif dist < 3.
        return -10
    end

    return 0
end


function isEndState(s1::ScenarioOne, s1state::ScenarioOneState)

    if s1state.uav_status == :crashed || s1state.uav_status == :landed
        return true
    end

    if s1state.aircraft_status == :left
        return true
    end

    return false
end


end


