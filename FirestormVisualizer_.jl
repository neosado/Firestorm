# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

module FirestormVisualizer_

export FirestormVisualizer, visInit, visUpdate, updateAnimation, saveAnimation


using Visualizer_
using Firestorm_

using PyCall
using PyPlot

@pyimport matplotlib.animation as ani

import Visualizer_.visInit
import Visualizer_.visUpdate
import Visualizer_.updateAnimation
import Visualizer_.saveAnimation


type FirestormVisualizer <: Visualizer

    fig::Union(Figure, Nothing)
    ax1::Union(PyObject, Nothing)
    ax2::Union(PyObject, Nothing)
    ax3::Union(PyObject, Nothing)
    ax4::Union(PyObject, Nothing)

    artists::Union(Vector{PyObject}, Nothing)

    ims::Vector{Any}

    wait::Bool


    function FirestormVisualizer(;wait = false)

        self = new()

        self.fig = nothing
        self.ax1 = nothing
        self.ax2 = nothing
        self.ax3 = nothing
        self.ax4 = nothing

        self.artists = nothing

        self.ims = {}

        self.wait = wait
        
        return self
    end
end


function visInit(fsv::FirestormVisualizer, fs::Firestorm)

    if fsv.fig == nothing
        fig = figure()
        fig[:suptitle]("UAS Surveillance")

        ax1 = fig[:add_subplot](221)
        ax1[:set_aspect]("equal")
        ax1[:invert_yaxis]()
        ax1[:set_xticks]([0.5:(fs.nrow - 1.5)])
        ax1[:set_yticks]([0.5:(fs.ncol - 1.5)])
        ax1[:set_xticklabels]([])
        ax1[:set_yticklabels]([])
        ax1[:grid](true)
        ax1[:set_title]("Wildfire")

        ax2 = fig[:add_subplot](222)
        ax2[:set_aspect]("equal")
        ax2[:invert_yaxis]()
        ax2[:set_xticks]([0.5:(fs.nrow - 1.5)])
        ax2[:set_yticks]([0.5:(fs.ncol - 1.5)])
        ax2[:set_xticklabels]([])
        ax2[:set_yticklabels]([])
        ax2[:grid](true)
        ax2[:set_title]("Fuel")

        ax3 = fig[:add_subplot](223)
        ax3[:set_aspect]("equal")
        ax3[:invert_yaxis]()
        ax3[:set_xticks]([0.5:(fs.nrow - 1.5)])
        ax3[:set_yticks]([0.5:(fs.ncol - 1.5)])
        ax3[:set_xticklabels]([])
        ax3[:set_yticklabels]([])
        ax3[:grid](true)
        ax3[:set_title]("Rewards")

        ax4 = fig[:add_subplot](224)
        ax4[:set_aspect]("equal")
        ax4[:invert_yaxis]()
        ax4[:set_xticks]([0.5:(fs.nrow - 1.5)])
        ax4[:set_yticks]([0.5:(fs.ncol - 1.5)])
        ax4[:set_xticklabels]([])
        ax4[:set_yticklabels]([])
        ax4[:grid](true)
        ax4[:set_title]("Utilities")

        fsv.fig = fig
        fsv.ax1 = ax1
        fsv.ax2 = ax2
        fsv.ax3 = ax3
        fsv.ax4 = ax4
    else
        fig = fsv.fig
        ax1 = fsv.ax1
        ax2 = fsv.ax2
        fsv.ax3 = ax3
        fsv.ax4 = ax4

        for artist in fsv.artists
            artist[:set_visible](false)
        end
    end

    artists = PyObject[]

    wildfire_map = ax1[:imshow](fs.B, cmap = "Reds", alpha = 0.5, interpolation = "none")
    push!(artists, wildfire_map)

    fuel_map = ax2[:imshow](fs.F, cmap = "Greens", alpha = 0.7, vmin = 0, vmax = fs.F_max, interpolation = "none")
    push!(artists, fuel_map)

    reward_map = ax3[:imshow](fs.R, cmap = "Greys_r", alpha = 0.7, vmin = fs.R_min, vmax = fs.R_max, interpolation = "none")
    push!(artists, reward_map)

    utility_map = ax4[:imshow](fs.U, cmap = "Greys_r", alpha = 0.7, vmin = fs.U_min, vmax = fs.U_max, interpolation = "none")
    push!(artists, utility_map)


    uav_marker = ax[:plot](fs.uav_pos[2], fs.uav_pos[1], "b^", markefsize = 100 / fs.n)
    append!(artists, uav_marker)


    fig[:canvas][:draw]()

    fsv.artists = artists

    return fsv
end


function visUpdate(fsv::FirestormVisualizer, fs::Firestorm)

    fig = fsv.fig

    text = fig[:text](0.5, 0.2, "$(fs.nrow) x $(fs.ncol) grid, seed: $(fs.seed)", horizontalalignment = "center", verticalalignment = "top")
    push!(fsv.artists, text)

    fig[:canvas][:draw]()

    return fsv
end


function visUpdate(fsv::FirestormVisualizer, fs::Firestorm, sim::(Int64, FSAction, FSObservation, Float64, Float64))

    fig = fsv.fig

    timestep = sim[1]
    action = sim[2].action
    observation = sim[3].observation
    r = sim[4]
    R = sim[5]

    text = fig[:text](0.5, 0.2, "timestep: $timestep, action: $action, observation: $observation, reward: $(int(r)), total reward: $(int(R))", horizontalalignment = "center", verticalalignment = "top")
    push!(fsv.artists, text)

    fig[:canvas][:draw]()

    return fsv
end


function updateAnimation(fsv::FirestormVisualizer)

    append!(fsv.ims, {fsv.artists})

    if fsv.wait
        readline()
    end
end


function saveAnimation(fsv::FirestormVisualizer; interval::Int64 = 1000, repeat::Bool = false, filename::ASCIIString = "firestorm.mp4")

    im_ani = ani.ArtistAnimation(fsv.fig, fsv.ims, interval = interval, repeat = repeat, repeat_delay = interval * 5)
    im_ani[:save](filename)

    if repeat || fsv.wait
        readline()
    end
end


end


