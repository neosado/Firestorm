# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/13/2015

module ScenarioOne_

export ScenarioOne, ScenarioOneParams, ScenarioOneState
export initTrajectories, updateStatePartA, getReward, updateStatePartB, isEndState


using Wildfire_

using Base.Test
using Distributions


typealias PATH  Array{Float64, 2}


type ScenarioOneParams

    seed::Union(Uint64, Nothing)

    n::Int64

    sim_time::Int64
    sim_time_mu::Float64
    sim_time_sigma::Float64

    wf_init_loc::Union((Int64, Int64), Nothing)
    wf_sim_time::Int64
    wf_p_fire::Float64

    p_uav_landing_crash_rate::Float64
    p_uav_lower_crash_rate::Float64
    #p_uav_fail::Float64
    #p_uav_autopilot_fail::Float64

    r_crashed::Int64
    r_dist::Union(Array{Int64, 2}, Nothing)

    uav_loc::Union((Int64, Int64), Nothing)
    uav_base_loc::Union((Int64, Int64), Nothing)
    uav_velocity::Float64
    uav_policy::Symbol

    aircraft_start_loc::Union((Int64, Int64), Nothing)
    aircraft_end_loc::Union((Int64, Int64), Nothing)
    aircraft_control_points::Union(Vector{(Float64, Float64)}, Nothing)
    aircraft_velocity::Float64
    aircraft_traj_uncertainty::Float64

    aircraft_traj_adaptive::Bool
    aircraft_operation_time_limit::Int64


    function ScenarioOneParams()

        self = new()

        self.seed = nothing

        self.n = 5

        self.sim_time = self.n * self.n
        self.sim_time_mu = 0.
        self.sim_time_sigma = 0.

        self.wf_init_loc = nothing
        self.wf_sim_time = self.n
        self.wf_p_fire = 0.06

        self.p_uav_landing_crash_rate = 0.
        self.p_uav_lower_crash_rate = 0.

        self.r_crashed = 0
        self.r_dist = nothing

        self.uav_loc = nothing
        self.uav_base_loc = nothing
        self.uav_velocity = 1.
        self.uav_policy = :back

        self.aircraft_start_loc = nothing
        self.aircraft_end_loc = nothing
        self.aircraft_control_points = nothing
        self.aircraft_velocity = 1.
        self.aircraft_traj_uncertainty = 0.

        self.aircraft_traj_adaptive = false
        self.aircraft_operation_time_limit = 0

        return self
    end
end


type ScenarioOne

    seed::Uint64

    nrow::Int64
    ncol::Int64

    wm::Wildfire


    sim_time::Int64
    sim_time_mu::Float64
    sim_time_sigma::Float64


    p_uav_landing_crash_rate::Float64
    p_uav_lower_crash_rate::Float64
    #p_uav_fail::Float64
    #p_uav_autopilot_fail::Float64


    r_crashed::Int64
    r_dist::Union(Array{Int64, 2}, Nothing)


    uav_loc::(Int64, Int64)
    uav_base_loc::(Int64, Int64)

    uav_velocity::Float64

    uav_policy::Symbol

    uav_path::Union(PATH, Nothing)
    uav_dpath::Vector{(Int64, Int64)}


    aircraft_start_loc::(Int64, Int64)
    aircraft_end_loc::(Int64, Int64)
    aircraft_control_points::Union(Vector{(Float64, Float64)}, Nothing)

    aircraft_velocity::Float64
    aircraft_traj_uncertainty::Float64

    aircraft_traj_adaptive::Bool
    aircraft_operation_time_limit::Int64

    aircraft_planned_path::PATH
    aircraft_planned_dpath::Vector{(Int64, Int64)}

    aircraft_path::PATH
    aircraft_dpath::Vector{(Int64, Int64)}


    function ScenarioOne(params::ScenarioOneParams)

        self = new()


        if params.seed != nothing
            if params.seed != 0
                self.seed = uint(params.seed)
            else
                self.seed = uint(time())
            end

            srand(self.seed)
        end


        nrow = params.n
        ncol = params.n

        self.nrow = nrow
        self.ncol = ncol


        self.wm = Wildfire(nrow, ncol, seed = params.seed, init_loc = params.wf_init_loc, p_fire = params.wf_p_fire)

        simulate_wildfire(self.wm, params.wf_sim_time)


        if params.sim_time_mu == 0.
            self.sim_time = params.sim_time
        elseif params.sim_time_sigma == 0.
            self.sim_time = params.sim_time_mu
        else
            self.sim_time = int(rand(TruncatedNormal(params.sim_time_mu, params.sim_time_sigma, 1., Inf)))
        end
        self.sim_time_mu = params.sim_time_mu
        self.sim_time_sigma = params.sim_time_sigma


        self.p_uav_landing_crash_rate = params.p_uav_landing_crash_rate
        self.p_uav_lower_crash_rate = params.p_uav_lower_crash_rate


        self.r_crashed = params.r_crashed
        self.r_dist = params.r_dist


        if params.uav_loc != nothing
            self.uav_loc = params.uav_loc
        else
            #self.uav_loc = (rand(1:nrow), rand(1:ncol))
            locs = getUAVInitLocs(self.wm)
            self.uav_loc = locs[rand(1:length(locs))]
        end


        self.uav_velocity = params.uav_velocity
        self.uav_policy = params.uav_policy


        if params.uav_base_loc != nothing
            self.uav_base_loc = params.uav_base_loc
        else
            Lot = [(1, j) for j = 1:ncol]
            append!(Lot, [(nrow, j) for j = 1:ncol])
            append!(Lot, [(i, 1) for i = 2:nrow-1])
            append!(Lot, [(i, ncol) for i = 2:nrow-1])
            nLot = length(Lot)

            self.uav_base_loc = Lot[rand(1:nLot)]
        end

        if params.aircraft_start_loc != nothing && params.aircraft_end_loc != nothing
            self.aircraft_start_loc = params.aircraft_start_loc
            self.aircraft_end_loc = params.aircraft_end_loc
        else
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
        end

        self.aircraft_control_points = params.aircraft_control_points

        self.aircraft_velocity = params.aircraft_velocity
        self.aircraft_traj_uncertainty = params.aircraft_traj_uncertainty

        self.aircraft_traj_adaptive = params.aircraft_traj_adaptive
        self.aircraft_operation_time_limit = params.aircraft_operation_time_limit


        return self
    end
end


type ScenarioOneState

    uav_loc::Union((Int64, Int64), Nothing)
    uav_status::Symbol

    aircraft_loc::Union((Int64, Int64), Nothing)
    aircraft_prev_loc::Union((Int64, Int64), Nothing)
    aircraft_status::Symbol
    aircraft_mode::Symbol


    function ScenarioOneState(s1::ScenarioOne)

        self = new()

        self.uav_loc = s1.uav_loc
        self.uav_status = :flying

        self.aircraft_loc = s1.aircraft_start_loc
        self.aircraft_prev_loc = nothing
        self.aircraft_status = :flying
        self.aircraft_mode = :search

        return self
    end
end


function simulate_wildfire(wm, ts_sim = 1; draw = false, wait = false)

    if draw
        wfv = WildfireVisualizer(wait = wait)

        visInit(wfv, wm)
        visUpdate(wfv, wm)
        updateAnimation(wfv)
    end

    for t = 1:ts_sim
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


function getUAVInitLocsHelper(M::BurningMatrix)

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


function generate_aircraft_trajectory(s1::ScenarioOne; bPerturb::Bool = true)

    if s1.aircraft_control_points == nothing
        P = [s1.aircraft_start_loc, s1.aircraft_end_loc]
    else
        if s1.aircraft_traj_uncertainty != 0. && bPerturb == true
            D = Normal(0., s1.aircraft_traj_uncertainty)

            CPs = (Float64, Float64)[]
            for cp in s1.aircraft_control_points
                push!(CPs, (cp[1] + rand(D), cp[2] + rand(D)))
            end
        else
            CPs = s1.aircraft_control_points
        end

        P = [s1.aircraft_start_loc, CPs, s1.aircraft_end_loc]
    end

    d = length(P) - 1

    nc = s1.nrow * s1.ncol

    path = zeros(nc, 2)

    k = 1
    for t = linspace(0, 1, nc)
        # Bezier curve using Bernstein basis polynomials
        # http://en.wikipedia.org/wiki/B챕zier_curve
        path[k, 1] = sum([factorial(d) / (factorial(i) * factorial(d-i)) * t^i * (1-t)^(d-i) * P[i+1][1] for i = 0:d])
        path[k, 2] = sum([factorial(d) / (factorial(i) * factorial(d-i)) * t^i * (1-t)^(d-i) * P[i+1][2] for i = 0:d])

        # straight line
        #path[k, 1] = t * s1.aircraft_end_loc[1] + (1 - t) * s1.aircraft_start_loc[1]
        #path[k, 2] = t * s1.aircraft_end_loc[2] + (1 - t) * s1.aircraft_start_loc[2]

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


function discretize_path(path::PATH, velocity)

    dpath = (Int64, Int64)[]
    vratio = int(1 / velocity)

    prev_row = int(path[1, 1])
    prev_col = int(path[1, 2])

    for j = 1:vratio
        push!(dpath, (prev_row, prev_col))
    end

    for i = 2:size(path, 1)
        curr_row = int(path[i, 1])
        curr_col = int(path[i, 2])

        drow = curr_row - prev_row
        dcol = curr_col - prev_col

        @assert abs(drow) <= 1 && abs(dcol) <= 1

        if abs(drow) == 1 && abs(dcol) == 1
            # be careful about this randomness
            if rand() < 0.5
                drow = 0
            else
                dcol = 0
            end

            for j = 1:vratio
                push!(dpath, (prev_row + drow, prev_col + dcol))
            end
        end

        if drow != 0 || dcol != 0
            for j = 1:vratio
                push!(dpath, (curr_row, curr_col))
            end

            prev_row = curr_row
            prev_col = curr_col
        end
    end

    return dpath
end


function initTrajectories(s1::ScenarioOne)

    s1.aircraft_path = generate_aircraft_trajectory(s1)
    s1.aircraft_dpath = discretize_path(s1.aircraft_path, s1.aircraft_velocity)

    if s1.aircraft_traj_uncertainty == 0.
        s1.aircraft_planned_path = s1.aircraft_path
        s1.aircraft_planned_dpath = s1.aircraft_dpath
    else
        s1.aircraft_planned_path = generate_aircraft_trajectory(s1, bPerturb = false)
        s1.aircraft_planned_dpath = discretize_path(s1.aircraft_planned_path, s1.uav_velocity)
    end

    s1.uav_path = generate_uav_trajectory(s1)
    if s1.uav_path != nothing
        s1.uav_dpath = discretize_path(s1.uav_path, s1.uav_velocity)
    end
end


function getNextUAVLoc(s1::ScenarioOne, t::Int64)

    if s1.uav_policy == :back
        if t + 1 > length(s1.uav_dpath)
            return nothing
        else
            return s1.uav_dpath[t + 1]
        end
    elseif s1.uav_policy == :landing || s1.uav_policy == :stay || s1.uav_policy == :lower
        return s1.uav_loc
    end
end


function isValidLoc(s1::ScenarioOne, loc::(Int64, Int64))

    if loc[1] == 0 || loc[1] == s1.nrow + 1 || loc[2] == 0 || loc[2] == s1.ncol + 1
        return false
    else
        return true
    end
end

function isValidNextAircraftMove(s1::ScenarioOne, aircraft_next_loc::(Int64, Int64))

    bOk = false

    for dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        aircraft_next_next_loc = (aircraft_next_loc[1] + dir[1], aircraft_next_loc[2] + dir[2])

        if isValidLoc(s1, aircraft_next_next_loc)
            if s1.wm.B[aircraft_next_loc...]
                if !s1.wm.B[aircraft_next_next_loc...]
                    bOk = true
                    break
                end
            else
                if s1.wm.B[aircraft_next_next_loc...]
                    bOk = true
                    break
                end
            end
        end
    end

    return bOk
end

function getNextAircraftLoc(s1::ScenarioOne, s1state::ScenarioOneState, t::Int64)

    if s1.aircraft_traj_adaptive
        aircraft_loc = s1state.aircraft_loc
        aircraft_prev_loc = s1state.aircraft_prev_loc

        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        if s1state.aircraft_mode == :search
            bSurveil = false

            for dir in dirs
                aircraft_next_loc = (aircraft_loc[1] + dir[1], aircraft_loc[2] + dir[2])
                if isValidLoc(s1, aircraft_next_loc) && s1.wm.B[aircraft_next_loc[1], aircraft_next_loc[2]]
                    bSurveil = true
                end
            end

            if bSurveil
                s1state.aircraft_mode = :surveil
            end

        elseif t == s1.aircraft_operation_time_limit
            s1state.aircraft_mode = :back
        end

        if s1state.aircraft_mode == :search
            if t + 1 > length(s1.aircraft_dpath)
                return nothing
            else
                return s1.aircraft_dpath[t + 1]
            end

        elseif s1state.aircraft_mode == :surveil
            validMove = (Int64, Int64)[]

            for dir in dirs
                aircraft_next_loc = (aircraft_loc[1] + dir[1], aircraft_loc[2] + dir[2])

                if isValidLoc(s1, aircraft_next_loc) && aircraft_next_loc != aircraft_prev_loc
                    if isValidNextAircraftMove(s1, aircraft_next_loc)
                        push!(validMove, aircraft_next_loc)
                    end
                end
            end

            if length(validMove) != 0
                return validMove[rand(1:length(validMove))]
            else
                for dir in dirs
                    aircraft_next_loc = (aircraft_loc[1] + dir[1], aircraft_loc[2] + dir[2])

                    if isValidLoc(s1, aircraft_next_loc) && aircraft_next_loc != aircraft_prev_loc
                        push!(validMove, aircraft_next_loc)
                    end
                end

                return validMove[rand(1:length(validMove))]
            end

        elseif s1state.aircraft_mode == :back
            if aircraft_loc == s1.aircraft_end_loc
                return nothing
            end

            dists = ones(Int64, 4) * Inf
            validMove = (Int64, Int64)[]

            for i = 1:length(dirs)
                aircraft_next_loc = (aircraft_loc[1] + dirs[i][1], aircraft_loc[2] + dirs[i][2])
                if isValidLoc(s1, aircraft_next_loc)
                    dists[i] = abs(aircraft_next_loc[1] - s1.aircraft_end_loc[1]) + abs(aircraft_next_loc[2] - s1.aircraft_end_loc[2])
                end
            end

            min_dist = minimum(dists)
            for i = 1:length(dirs)
                aircraft_next_loc = (aircraft_loc[1] + dirs[i][1], aircraft_loc[2] + dirs[i][2])

                if dists[i] == min_dist
                    push!(validMove, aircraft_next_loc)
                end
            end

            return validMove[rand(1:length(validMove))]
        end

    else
        if t + 1 > length(s1.aircraft_dpath)
            return nothing
        else
            return s1.aircraft_dpath[t + 1]
        end
    end
end


function updateStatePartA(s1::ScenarioOne, s1state::ScenarioOneState, t::Int64)

    s1state.uav_loc = getNextUAVLoc(s1, t)

    aircraft_next_loc = getNextAircraftLoc(s1, s1state, t)
    s1state.aircraft_prev_loc = s1state.aircraft_loc
    s1state.aircraft_loc = aircraft_next_loc

    if t > 0 && s1.uav_policy == :landing
        if s1.wm.B[s1state.uav_loc[1], s1state.uav_loc[2]]
            s1state.uav_status = :crashed
        else
            if rand() < s1.p_uav_landing_crash_rate
                s1state.uav_status = :crashed
            else
                s1state.uav_status = :landed
            end
        end
    elseif t > 0 && s1.uav_policy == :lower
        if rand() < s1.p_uav_lower_crash_rate
            s1state.uav_status = :crashed
        end
    end
end


function getReward(s1::ScenarioOne, s1state::ScenarioOneState, t::Int64)

    if t == 0
        return 0
    end

    if s1state.uav_status == :landed
        return 0
    end

    if s1state.uav_status == :crashed
        return s1.r_crashed
    end

    if s1.uav_policy != :lower && s1.r_dist != nothing
        if s1state.uav_loc != nothing && s1state.aircraft_loc != nothing
            #dist = norm([s1state.uav_loc[1] - s1state.aircraft_loc[1], s1state.uav_loc[2] - s1state.aircraft_loc[2]])
            dist = abs(s1state.uav_loc[1] - s1state.aircraft_loc[1]) + abs(s1state.uav_loc[2] - s1state.aircraft_loc[2])

            for i = 1:size(s1.r_dist, 1)
                if dist < s1.r_dist[i, 1]
                    return s1.r_dist[i, 2]
                end
            end
        end
    end

    return 0
end


function updateStatePartB(s1::ScenarioOne, s1state::ScenarioOneState, t::Int64)

    if s1state.uav_loc == nothing
        s1state.uav_status = :left
    end

    if s1state.aircraft_loc == nothing
        s1state.aircraft_status = :left
    end

    if t > 0 && s1.uav_policy != :lower && s1state.uav_loc == s1state.aircraft_loc
        s1state.uav_status = :collided
        s1state.aircraft_status = :collided
    end

    wfNextState(s1.wm)
end


function isEndState(s1::ScenarioOne, s1state::ScenarioOneState, t::Int64)

    if t > s1.sim_time
        return true
    end

    if s1state.uav_status == :left || s1state.uav_status == :crashed || s1state.uav_status == :landed
        return true
    end

    if s1state.aircraft_status == :left
        return true
    end

    if s1state.uav_status == :collided || s1state.aircraft_status == :collided
        return true
    end

    return false
end


end


