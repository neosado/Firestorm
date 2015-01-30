# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/13/2015

module ScenarioOneVisualizer_

export ScenarioOneVisualizer, visInit, visUpdate, updateAnimation, saveAnimation


using Visualizer_
using ScenarioOne_

using PyCall
using PyPlot

@pyimport matplotlib.animation as ani

import Visualizer_.visInit
import Visualizer_.visUpdate
import Visualizer_.updateAnimation
import Visualizer_.saveAnimation


type ScenarioOneVisualizer <: Visualizer

    fig::Union(Figure, Nothing)
    ax1::Union(PyObject, Nothing)

    artists::Union(Vector{PyObject}, Nothing)

    ims::Vector{Any}

    wait::Bool


    function ScenarioOneVisualizer(;wait = false)

        self = new()

        self.fig = nothing
        self.ax1 = nothing

        self.artists = nothing

        self.ims = {}

        self.wait = wait
        
        return self
    end
end


function visInit(s1v::ScenarioOneVisualizer, s1::ScenarioOne, timestep::Int64 = 0)

    if s1v.fig == nothing
        fig = figure(facecolor = "white")

        ax1 = fig[:add_subplot](111)
        ax1[:set_aspect]("equal")
        ax1[:invert_yaxis]()
        ax1[:set_xticks]([0.5:(s1.nrow - 1.5)])
        ax1[:set_yticks]([0.5:(s1.ncol - 1.5)])
        ax1[:set_xticklabels]([])
        ax1[:set_yticklabels]([])
        ax1[:grid](true)
        ax1[:set_title]("ScenarioOne")

        s1v.fig = fig
        s1v.ax1 = ax1
    else
        fig = s1v.fig
        ax1 = s1v.ax1

        for artist in s1v.artists
            artist[:set_visible](false)
        end
    end

    artists = PyObject[]


    wildfire_map = ax1[:imshow](s1.wm.B, cmap = "Reds", alpha = 0.5, interpolation = "none")
    push!(artists, wildfire_map)


    if s1.aircraft_traj_uncertainty != 0
        #aircraft_planned_path = ax1[:plot](s1.aircraft_planned_path[:, 2] - 1, s1.aircraft_planned_path[:, 1] - 1, linestyle = "--", color = "0.7")
        path = zeros(Int64, length(s1.aircraft_planned_dpath), 2)
        for t = 1:length(s1.aircraft_planned_dpath)
            path[t, 1], path[t, 2] = s1.aircraft_planned_dpath[t]
        end
        aircraft_planned_path = ax1[:plot](path[:, 2] - 1, path[:, 1] - 1, "--", color = "0.7")
        append!(artists, aircraft_planned_path)
    end

    #aircraft_path = ax1[:plot](s1.aircraft_path[:, 2] - 1, s1.aircraft_path[:, 1] - 1, "c--")
    path = zeros(Int64, length(s1.aircraft_dpath), 2)
    for t = 1:length(s1.aircraft_dpath)
        path[t, 1], path[t, 2] = s1.aircraft_dpath[t]
    end
    aircraft_path = ax1[:plot](path[:, 2] - 1, path[:, 1] - 1, "c--")
    append!(artists, aircraft_path)

    aircraft_start_loc = ax1[:plot](s1.aircraft_path[1, 2] - 1, s1.aircraft_path[1, 1] - 1, "k.")
    append!(artists, aircraft_start_loc)


    if s1.uav_path != nothing
        #uav_path = ax1[:plot](s1.uav_path[:, 2] - 1, s1.uav_path[:, 1] - 1, "r-.")
        path = zeros(Int64, length(s1.uav_dpath), 2)
        for t = 1:length(s1.uav_dpath)
            path[t, 1], path[t, 2] = s1.uav_dpath[t]
        end
        uav_path = ax1[:plot](path[:, 2] - 1, path[:, 1] - 1, "r-.")
        append!(artists, uav_path)

        uav_start_loc = ax1[:plot](s1.uav_path[1, 2] - 1, s1.uav_path[1, 1] - 1, "k.")
        append!(artists, uav_start_loc)
    end

    uav_base_loc = ax1[:plot](s1.uav_base_loc[2] - 1, s1.uav_base_loc[1] - 1, "kx")
    append!(artists, uav_base_loc)


    fig[:canvas][:draw]()

    s1v.artists = artists

    return s1v
end


function visUpdate(s1v::ScenarioOneVisualizer, s1::ScenarioOne)

    fig = s1v.fig
    ax1 = s1v.ax1


    text = s1v.ax1[:text](0.5, -0.02, "$(s1.nrow) x $(s1.ncol) grid, seed: $(s1.seed)", horizontalalignment = "center", verticalalignment = "top", transform = s1v.ax1[:transAxes])
    push!(s1v.artists, text)


    aircraft_marker = ax1[:plot](s1.aircraft_start_loc[2] - 1, s1.aircraft_start_loc[1] - 1, "b^", markersize = 150 / min(s1.nrow, s1.ncol))
    append!(s1v.artists, aircraft_marker)


    uav_marker = ax1[:plot](s1.uav_loc[2] - 1, s1.uav_loc[1] - 1, "mo", markersize = 150 / min(s1.nrow, s1.ncol))
    append!(s1v.artists, uav_marker)


    fig[:canvas][:draw]()

    return s1v
end


function visUpdate(s1v::ScenarioOneVisualizer, s1::ScenarioOne, timestep::Int64, s1state::ScenarioOneState, reward::Int64, utility::Int64)

    fig = s1v.fig
    ax1 = s1v.ax1


    text = s1v.ax1[:text](0.5, -0.02, "timestep: $timestep, reward: $reward, utility: $utility", horizontalalignment = "center", verticalalignment = "top", transform = s1v.ax1[:transAxes])
    push!(s1v.artists, text)


    if s1state.uav_loc != nothing
        if s1state.uav_status == :flying
            uav_marker_spec = "mo"
        elseif s1state.uav_status == :landed
            uav_marker_spec = "go"
        elseif s1state.uav_status == :crashed
            uav_marker_spec = "ko"
        end

        uav_marker = ax1[:plot](s1state.uav_loc[2] - 1, s1state.uav_loc[1] - 1, uav_marker_spec, markersize = 150 / min(s1.nrow, s1.ncol))
        append!(s1v.artists, uav_marker)
    end


    if s1state.aircraft_loc != nothing
        aircraft_marker = ax1[:plot](s1state.aircraft_loc[2] - 1, s1state.aircraft_loc[1] - 1, "b^", markersize = 150 / min(s1.nrow, s1.ncol))
        append!(s1v.artists, aircraft_marker)
    end


    fig[:canvas][:draw]()

    return s1v
end


function updateAnimation(s1v::ScenarioOneVisualizer, timestep::Int64 = -1; bSaveFrame::Bool = false, filename::ASCIIString = "scenario1.png")

    append!(s1v.ims, {s1v.artists})

    if bSaveFrame
        if timestep == -1
            savefig(filename, transparent = false)
        else
            base, ext = splitext(filename)
            savefig(base * "_" * string(timestep) * "." * ext, transparent = false)
        end
    end

    if s1v.wait
        readline()
    end
end


function saveAnimation(s1v::ScenarioOneVisualizer; interval::Int64 = 1000, repeat::Bool = false, filename::ASCIIString = "scenario1.mp4")

    im_ani = ani.ArtistAnimation(s1v.fig, s1v.ims, interval = interval, repeat = repeat, repeat_delay = interval * 5)
    im_ani[:save](filename)

    if repeat || s1v.wait
        readline()
    end
end


end


