# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/04/2014

module FirestormVisualizer_

export FirestormVisualizer, visInit, visUpdate, updateAnimation, saveAnimation


using Visualizer_
using Firestorm_
using Wildfire_

using Base.Test
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


function visInit(fsv::FirestormVisualizer, fs::Firestorm, b::FSBelief)

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
        ax4[:set_title]("Belief")

        fsv.fig = fig
        fsv.ax1 = ax1
        fsv.ax2 = ax2
        fsv.ax3 = ax3
        fsv.ax4 = ax4
    else
        fig = fsv.fig
        ax1 = fsv.ax1
        ax2 = fsv.ax2
        ax3 = fsv.ax3
        ax4 = fsv.ax4

        for artist in fsv.artists
            artist[:set_visible](false)
        end
    end

    artists = PyObject[]

    wildfire_map = ax1[:imshow](fs.wm.B, cmap = "Reds", alpha = 0.5, interpolation = "none")
    push!(artists, wildfire_map)

    fuel_map = ax2[:imshow](fs.wm.F, cmap = "Greens", alpha = 0.7, vmin = 0, vmax = fs.wm.F_max, interpolation = "none")
    push!(artists, fuel_map)


    P = burningProbMatrix(fs.wm)
    P[fs.wm.B] = 0.
    R = P
    R_min = 0.
    R_max = 1.

    reward_map = ax3[:imshow](R, cmap = "Greys", alpha = 0.7, vmin = R_min, vmax = R_max, interpolation = "none")
    push!(artists, reward_map)


    k_max = nothing
    v_max = 0.
    if typeof(b) == FSBeliefVector
        for (k, v) in b.belief
            if k.Position != fs.uav_pos
                if v != 0.
                    println(k)
                    println(v)
                    exit(1)
                end
            end

            if v_max != 0. && v == v_max
                if rand() < 0.5
                    k_max = k
                    v_max = v
                end
            elseif v > v_max
                k_max = k
                v_max = v
            end
        end
    elseif typeof(b) == FSBeliefParticles
        count_ = Dict{FSState, Int64}()

        for s in b.particles
            if haskey(count_, s)
                count_[s] += 1
            else
                count_[s] = 1
            end
        end

        for (k, v) in count_
            if k.Position != fs.uav_pos
                @test v == 0.
            end

            if v_max != 0. && v == v_max
                if rand() < 0.5
                    k_max = k
                    v_max = v
                end
            elseif v > v_max
                k_max = k
                v_max = v
            end
        end
    end

    Bf = k_max.B
    Bf_min = 0.
    Bf_max = 1.

    utility_map = ax4[:imshow](Bf, cmap = "Reds", alpha = 0.5, vmin = Bf_min, vmax = Bf_max, interpolation = "none")
    push!(artists, utility_map)

    # TODO add colorbar


    uav_marker = ax1[:plot](fs.uav_pos[2] - 1, fs.uav_pos[1] - 1, "b^", markersize = 50 / min(fs.nrow, fs.ncol))
    append!(artists, uav_marker)


    fig[:canvas][:draw]()

    fsv.artists = artists

    return fsv
end


function visUpdate(fsv::FirestormVisualizer, fs::Firestorm)

    fig = fsv.fig

    # FIXME ArtistAnimation fails with fig. ax works ok.
    #text = fig[:text](0.5, 0.065, "$(fs.nrow) x $(fs.ncol) grid, seed: $(fs.seed)", horizontalalignment = "center", verticalalignment = "top")
    text = fsv.ax3[:text](1.25, -0.1, "$(fs.nrow) x $(fs.ncol) grid, seed: $(fs.seed)", horizontalalignment = "center", verticalalignment = "top", transform = fsv.ax3[:transAxes])
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

    # FIXME ArtistAnimation fails with fig. ax works ok.
    #text = fig[:text](0.5, 0.065, "timestep: $timestep, action: $action, observation: $observation, reward: $(int(r)), total reward: $(int(R))", horizontalalignment = "center", verticalalignment = "top")
    text = fsv.ax3[:text](1.25, -0.1, "timestep: $timestep, action: $action, observation: $observation, reward: $(int(r)), total reward: $(int(R))", horizontalalignment = "center", verticalalignment = "top", transform = fsv.ax3[:transAxes])
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


