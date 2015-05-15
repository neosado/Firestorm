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

    plots::Union(Symbol, Vector{Symbol})
    nplots::Int64

    fig::Union(Figure, Nothing)
    ax::Dict{Symbol, PyObject}

    artists::Union(Vector{PyObject}, Nothing)

    ims::Vector{Any}

    wait::Bool

    b_label::Bool

    initialized::Bool


    function WildfireVisualizer(; plots::Union(Symbol, Vector{Symbol}) = :all, wait::Bool = false, bLabel::Bool = true)

        self = new()

        if plots == :all
            self.plots = [:wildfire, :fuel, :propagation]
        elseif typeof(plots) == Symbol
            self.plots = [plots]
        else
            self.plots = plots
        end
        self.nplots = length(self.plots)

        self.fig = nothing

        self.ax = Dict{Symbol, PyObject}()

        self.artists = nothing

        self.ims = {}

        self.wait = wait

        self.b_label = bLabel

        self.initialized = false
        
        return self
    end
end


function visInit(wfv::WildfireVisualizer, wm::Wildfire)

    if wfv.fig == nothing
        wfv.fig = figure(facecolor = "white")

        if wfv.nplots == 1
            plot_base = 110
        elseif wfv.nplots == 2
            plot_base = 120
        else
            plot_base = 220
        end

        for i = 1:wfv.nplots
            pl = wfv.plots[i]

            ax = wfv.fig[:add_subplot](plot_base + i)
            ax[:set_aspect]("equal")
            ax[:invert_yaxis]()
            ax[:set_xticks]([0.5:(wm.nrow - 1.5)])
            ax[:set_yticks]([0.5:(wm.ncol - 1.5)])
            ax[:set_xticklabels]([])
            ax[:set_yticklabels]([])
            ax[:grid](true)

            if wfv.b_label
                if pl == :wildfire
                    ax[:set_title]("Wildfire")
                elseif pl == :fuel
                    ax[:set_title]("Fuel")
                elseif pl == :propagation
                    ax[:set_title]("Propagation")
                end
            end

            wfv.ax[pl] = ax
        end

    else
        for artist in wfv.artists
            artist[:set_visible](false)
        end

    end

    artists = PyObject[]

    if :wildfire in wfv.plots
        wildfire_map = wfv.ax[:wildfire][:imshow](wm.B, cmap = "Reds", alpha = 0.5, interpolation = "none")
        push!(artists, wildfire_map)
    end

    if :fuel in wfv.plots
        fuel_map = wfv.ax[:fuel][:imshow](wm.F, cmap = "Greens", alpha = 0.7, vmin = 0, vmax = wm.F_max, interpolation = "none")
        push!(artists, fuel_map)

        if wfv.initialized == false && wfv.nplots == 1
            cbar = colorbar(fuel_map)
        end
    end

    if :propagation in wfv.plots
        P = burningProbMatrix(wm)
        P[wm.B] = 0.

        prob_map = wfv.ax[:propagation][:imshow](P, cmap = "Greys", alpha = 0.7, vmin = 0., vmax = 1., interpolation = "none")
        push!(artists, prob_map)

        if wfv.initialized == false && wfv.nplots == 1
            cbar = colorbar(prob_map)
        end
    end

    wfv.fig[:canvas][:draw]()

    wfv.artists = artists

    wfv.initialized = true

    return wfv
end


function visUpdate(wfv::WildfireVisualizer, wm::Wildfire)

    if wfv.b_label
        # FIXME ArtistAnimation fails with fig. ax works ok.
        #text = wfv.fig[:text](0.5, 0.2, "$(wm.nrow) x $(wm.ncol) grid, seed: $(wm.seed)", horizontalalignment = "center", verticalalignment = "top")

        if wfv.nplots == 1
            pl = wfv.plots[1]
            x = 0.5
            y = -0.05
        elseif wfv.nplots == 2
            pl = wfv.plots[1]
            x = 1.1
            y = -0.1
        else
            pl = wfv.plots[3]
            x = 1.25
            y = -0.1
        end

        text = wfv.ax[pl][:text](x, y, "$(wm.nrow) x $(wm.ncol) grid, seed: $(wm.seed)", horizontalalignment = "center", verticalalignment = "top", transform = wfv.ax[pl][:transAxes])
        push!(wfv.artists, text)
    end

    wfv.fig[:canvas][:draw]()

    return wfv
end


function visUpdate(wfv::WildfireVisualizer, wm::Wildfire, timestep::Int64)

    if wfv.b_label
        # FIXME ArtistAnimation fails with fig. ax works ok.
        #text = wfv.fig[:text](0.5, 0.2, "timestep: $timestep", horizontalalignment = "center", verticalalignment = "top")

        if wfv.nplots == 1
            pl = wfv.plots[1]
            x = 0.5
            y = -0.05
        elseif wfv.nplots == 2
            pl = wfv.plots[1]
            x = 1.1
            y = -0.1
        else
            pl = wfv.plots[3]
            x = 1.25
            y = -0.1
        end

        text = wfv.ax[pl][:text](x, y, "timestep: $timestep", horizontalalignment = "center", verticalalignment = "top", transform = wfv.ax[pl][:transAxes])
        push!(wfv.artists, text)
    end

    wfv.fig[:canvas][:draw]()

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


