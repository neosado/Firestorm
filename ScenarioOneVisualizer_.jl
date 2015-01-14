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


function visInit(s1v::ScenarioOneVisualizer, s1::ScenarioOne)

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

    aircraft_path = ax1[:plot](s1.aircraft_path[:, 2] - 1, s1.aircraft_path[:, 1] - 1, "c--")
    append!(artists, aircraft_path)

    aircraft_start_loc = ax1[:plot](s1.aircraft_path[1, 2] - 1, s1.aircraft_path[1, 1] - 1, "k.")
    append!(artists, aircraft_start_loc)

    uav_path = ax1[:plot](s1.uav_path[:, 2] - 1, s1.uav_path[:, 1] - 1, "r-.")
    append!(artists, uav_path)

    uav_start_loc = ax1[:plot](s1.uav_path[1, 2] - 1, s1.uav_path[1, 1] - 1, "k.")
    append!(artists, uav_start_loc)

    uav_base_marker = ax1[:plot](s1.uav_base_loc[2] - 1, s1.uav_base_loc[1] - 1, "k+", markersize = 150 / min(s1.nrow, s1.ncol))
    append!(artists, uav_base_marker)


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

    if s1state.aircraft_status == :flying
        aircraft_marker = ax1[:plot](s1state.aircraft_loc[2] - 1, s1state.aircraft_loc[1] - 1, "b^", markersize = 150 / min(s1.nrow, s1.ncol))
        append!(s1v.artists, aircraft_marker)
    end

    if s1state.uav_status == :flying
        uav_marker = ax1[:plot](s1state.uav_loc[2] - 1, s1state.uav_loc[1] - 1, "mo", markersize = 150 / min(s1.nrow, s1.ncol))
        append!(s1v.artists, uav_marker)
    end

    fig[:canvas][:draw]()

    return s1v
end


function updateAnimation(s1v::ScenarioOneVisualizer)

    append!(s1v.ims, {s1v.artists})

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


