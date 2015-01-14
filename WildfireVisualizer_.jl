# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

module WildfireVisualizer_

export WildfireVisualizer, visInit, visUpdate, updateAnimation, saveAnimation


using Visualizer_
using Wildfire_

using PyCall
using PyPlot

@pyimport matplotlib.animation as ani

import Visualizer_.visInit
import Visualizer_.visUpdate
import Visualizer_.updateAnimation
import Visualizer_.saveAnimation


type WildfireVisualizer <: Visualizer

    fig::Union(Figure, Nothing)
    ax1::Union(PyObject, Nothing)
    ax2::Union(PyObject, Nothing)
    ax3::Union(PyObject, Nothing)

    artists::Union(Vector{PyObject}, Nothing)

    ims::Vector{Any}

    wait::Bool


    function WildfireVisualizer(;wait = false)

        self = new()

        self.fig = nothing
        self.ax1 = nothing
        self.ax2 = nothing
        self.ax3 = nothing

        self.artists = nothing

        self.ims = {}

        self.wait = wait
        
        return self
    end
end


function visInit(wfv::WildfireVisualizer, wm::Wildfire)

    if wfv.fig == nothing
        fig = figure(facecolor = "white")

        ax1 = fig[:add_subplot](221)
        ax1[:set_aspect]("equal")
        ax1[:invert_yaxis]()
        ax1[:set_xticks]([0.5:(wm.nrow - 1.5)])
        ax1[:set_yticks]([0.5:(wm.ncol - 1.5)])
        ax1[:set_xticklabels]([])
        ax1[:set_yticklabels]([])
        ax1[:grid](true)
        ax1[:set_title]("Wildfire")

        ax2 = fig[:add_subplot](222)
        ax2[:set_aspect]("equal")
        ax2[:invert_yaxis]()
        ax2[:set_xticks]([0.5:(wm.nrow - 1.5)])
        ax2[:set_yticks]([0.5:(wm.ncol - 1.5)])
        ax2[:set_xticklabels]([])
        ax2[:set_yticklabels]([])
        ax2[:grid](true)
        ax2[:set_title]("Fuel")

        ax3 = fig[:add_subplot](223)
        ax3[:set_aspect]("equal")
        ax3[:invert_yaxis]()
        ax3[:set_xticks]([0.5:(wm.nrow - 1.5)])
        ax3[:set_yticks]([0.5:(wm.ncol - 1.5)])
        ax3[:set_xticklabels]([])
        ax3[:set_yticklabels]([])
        ax3[:grid](true)
        ax3[:set_title]("Propagation")

        wfv.fig = fig
        wfv.ax1 = ax1
        wfv.ax2 = ax2
        wfv.ax3 = ax3
    else
        fig = wfv.fig
        ax1 = wfv.ax1
        ax2 = wfv.ax2
        ax3 = wfv.ax3

        for artist in wfv.artists
            artist[:set_visible](false)
        end
    end

    artists = PyObject[]

    wildfire_map = ax1[:imshow](wm.B, cmap = "Reds", alpha = 0.5, interpolation = "none")
    push!(artists, wildfire_map)

    fuel_map = ax2[:imshow](wm.F, cmap = "Greens", alpha = 0.7, vmin = 0, vmax = wm.F_max, interpolation = "none")
    push!(artists, fuel_map)

    P = burningProbMatrix(wm)
    P[wm.B] = 0.

    prob_map = ax3[:imshow](P, cmap = "Greys", alpha = 0.7, vmin = 0., vmax = 1., interpolation = "none")
    push!(artists, prob_map)

    # TODO add colorbar


    fig[:canvas][:draw]()

    wfv.artists = artists

    return wfv
end


function visUpdate(wfv::WildfireVisualizer, wm::Wildfire)

    fig = wfv.fig

    # FIXME ArtistAnimation fails with fig. ax works ok.
    #text = fig[:text](0.5, 0.2, "$(wm.nrow) x $(wm.ncol) grid, seed: $(wm.seed)", horizontalalignment = "center", verticalalignment = "top")
    text = wfv.ax3[:text](1.25, -0.1, "$(wm.nrow) x $(wm.ncol) grid, seed: $(wm.seed)", horizontalalignment = "center", verticalalignment = "top", transform = wfv.ax3[:transAxes])
    push!(wfv.artists, text)

    fig[:canvas][:draw]()

    return wfv
end


function visUpdate(wfv::WildfireVisualizer, wm::Wildfire, timestep::Int64)

    fig = wfv.fig

    # FIXME ArtistAnimation fails with fig. ax works ok.
    #text = fig[:text](0.5, 0.2, "timestep: $timestep", horizontalalignment = "center", verticalalignment = "top")
    text = wfv.ax3[:text](1.25, -0.1, "timestep: $timestep", horizontalalignment = "center", verticalalignment = "top", transform = wfv.ax3[:transAxes])
    push!(wfv.artists, text)

    fig[:canvas][:draw]()

    return wfv
end


function updateAnimation(wfv::WildfireVisualizer)

    append!(wfv.ims, {wfv.artists})

    if wfv.wait
        readline()
    end
end


function saveAnimation(wfv::WildfireVisualizer; interval::Int64 = 1000, repeat::Bool = false, filename::ASCIIString = "wildfire.mp4")

    im_ani = ani.ArtistAnimation(wfv.fig, wfv.ims, interval = interval, repeat = repeat, repeat_delay = interval * 5)
    im_ani[:save](filename)

    if repeat || wfv.wait
        readline()
    end
end


end


