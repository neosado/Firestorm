# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

module RockSampleVisualizer_

export RockSampleVisualizer, visInit, visUpdate, updateAnimation, saveAnimation


using Visualizer_
using RockSample_

using PyCall
using PyPlot

@pyimport matplotlib.animation as ani

import Visualizer_.visInit
import Visualizer_.visUpdate
import Visualizer_.updateAnimation
import Visualizer_.saveAnimation


type RockSampleVisualizer

    fig::Union(Figure, Nothing)
    ax::Union(PyObject, Nothing)
    artists::Union(Vector{PyObject}, Nothing)

    ims::Vector{Any}

    wait::Bool


    function RockSampleVisualizer(;wait = false)

        self = new()

        self.fig = nothing
        self.ax = nothing
        self.artists = nothing

        self.ims = {}

        self.wait = wait
        
        return self
    end
end


function visInit(rsv::RockSampleVisualizer, rs::RockSample)

    if rsv.fig == nothing
        fig = figure()

        ax = fig[:add_subplot](111)
        ax[:set_aspect]("equal")
        ax[:set_xlim](0.5, rs.n + 1.5)
        ax[:set_ylim](0.5, rs.n + 0.5)
        ax[:invert_yaxis]()
        ax[:set_xticks]([0.5:(rs.n + 0.5)])
        ax[:set_yticks]([0.5:(rs.n + 0.5)])
        ax[:set_xticklabels]([])
        ax[:set_yticklabels]([])
        ax[:grid](true)
        ax[:fill_between]([5.5, 6.5], rs.n + 0.5, 0.5, color = "k", alpha = 0.1)
        ax[:axvline](rs.n + 0.5, color = "k", linestyle = "--", alpha = 0.7)
        ax[:set_title]("RockSample")

        rsv.fig = fig
        rsv.ax = ax
    else
        fig = rsv.fig
        ax = rsv.ax

        for artist in rsv.artists
            artist[:set_visible](false)
        end
    end

    artists = PyObject[]

    for (rock, rock_type) in rs.rock_types
        rock_loc = rs.rock_locs[rock]
        rock_index = rock2ind(rock)

        if rock_type == :good
            spec = "^g"
        elseif rock_type == :bad
            spec = "^r"
        end

        if rs.rock_sampled[rock]
            alpha = 1.0
        else
            alpha = 0.3
        end

        rock_marker = ax[:plot](rock_loc[2], rock_loc[1], spec, markersize = 20, alpha = alpha)
        append!(artists, rock_marker)

        text = ax[:text](rock_loc[2], rock_loc[1], "$rock_index", horizontalalignment = "center", verticalalignment = "top")
        push!(artists, text)
    end

    rover_marker = ax[:plot](rs.rover_pos[2], rs.rover_pos[1], "sm", markersize = 20)
    append!(artists, rover_marker)

    fig[:canvas][:draw]()

    rsv.artists = artists

    return rsv
end


function visUpdate(rsv::RockSampleVisualizer, rs::RockSample)

    fig = rsv.fig
    ax = rsv.ax

    text = ax[:text](0.5, -0.02, "n: $(rs.n), k: $(rs.k), seed: $(rs.seed)", horizontalalignment = "center", verticalalignment = "top", transform = ax[:transAxes])
    push!(rsv.artists, text)

    fig[:canvas][:draw]()

    return rsv
end


function visUpdate(rsv::RockSampleVisualizer, rs::RockSample, sim::(Int64, RSAction, RSObservation, Float64, Float64))

    fig = rsv.fig
    ax = rsv.ax

    timestep = sim[1]
    action = sim[2].action
    observation = sim[3].observation
    r = sim[4]
    R = sim[5]

    text = ax[:text](0.5, -0.02, "timestep: $timestep, action: $action, observation: $observation, reward: $(int(r)), total reward: $(int(R))", horizontalalignment = "center", verticalalignment = "top", transform = ax[:transAxes])
    push!(rsv.artists, text)

    fig[:canvas][:draw]()

    return rsv
end


function updateAnimation(rsv::RockSampleVisualizer)

    append!(rsv.ims, {rsv.artists})

    if rsv.wait
        readline()
    end
end


function saveAnimation(rsv::RockSampleVisualizer; interval::Int64 = 1000, repeat::Bool = false, filename::ASCIIString = "rocksample.mp4")

    im_ani = ani.ArtistAnimation(rsv.fig, rsv.ims, interval = interval, repeat = repeat, repeat_delay = interval * 5)
    im_ani[:save](filename)

    if repeat || rsv.wait
        readline()
    end
end


end


