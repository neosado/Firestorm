# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/13/2015

using PyPlot
using HDF5, JLD


include("simScenarioOne.jl")


function retrieveEvaluation(param_set_num::Int64, policy::Symbol, uncertainty::Float64; database::ASCIIString = "s1results.jld", update::Bool = false, N_min::Int = 100, N_max::Int = 1000, RE_threshold::Float64 = 0.01, bParallel::Bool = false)

    if isfile(database)
        database = load(database, "DATA")
    else
        database = Dict{Uint64, Any}()
    end

    key = hash([param_set_num, policy, uncertainty])

    if update != true && haskey(database, key)
        params = database[key]["params"]
        U = database[key]["U"]
        RE = database[key]["RE"]
        N = database[key]["N"]
    else
        U, RE, N, params = evaluatePolicy(param_set_num, policy, uncertainty, N_min = N_min, N_max = N_max, RE_threshold = RE_threshold, bParallel = bParallel)
        
        record = Dict{ASCIIString, Any}()
        record["params"] = params
        record["U"] = U
        record["RE"] = RE
        record["N"] = N
        database[key] = record

        save("s1results.jld", "DATA", database)
    end
    
    # debug
    #println(params)
    #println(int(U))
    #println(round(RE, 4))
    #println(N)
    
    return U, RE, N, params
end


function plotEvaluation(param_set_num::Int64, policy::Symbol, uncertainty::Float64; draw::Bool = true, fig = nothing, database::ASCIIString = "s1results.jld")
    
    U, RE, N, params = retrieveEvaluation(param_set_num, policy, uncertainty, database = database)

    if draw
        if fig == nothing
            fig = figure(facecolor = "white")
        end

        ax1 = fig[:add_subplot](221)
        ax1[:set_xlim](-0.5, 10.5)
        ax1[:set_ylim](-0.5, 10.5)
        ax1[:set_aspect]("equal")
        ax1[:invert_yaxis]()
        ax1[:set_xticks]([0.5:(params.n - 1.5)])
        ax1[:set_yticks]([0.5:(params.n - 1.5)])
        ax1[:set_xticklabels]([])
        ax1[:set_yticklabels]([])
        ax1[:grid](true)
        ax1[:set_title]("Scenario")

        params.uav_loc = (4, 5)
        s1 = ScenarioOne(params)
        initTrajectories(s1)

        B = zeros(Bool, params.n, params.n)
        B[params.wf_init_loc[1], params.wf_init_loc[2]] = true
        ax1[:imshow](B, cmap = "Reds", alpha = 0.5, interpolation = "none")
        #ax1[:plot](params.wf_init_loc[2] - 1, params.wf_init_loc[1] - 1, "rs", alpha = 0.5, markersize = 50 / params.n)

        #ax1[:plot](s1.aircraft_path[:, 2] - 1, s1.aircraft_path[:, 1] - 1, "c--")
        path = zeros(Int64, length(s1.aircraft_dpath), 2)
        for t = 1:length(s1.aircraft_dpath)
            path[t, 1], path[t, 2] = s1.aircraft_dpath[t]
        end
        ax1[:plot](path[:, 2] - 1, path[:, 1] - 1, "c--")

        #ax1[:plot](s1.aircraft_planned_path[:, 2] - 1, s1.aircraft_planned_path[:, 1] - 1, linestyle = "--", color = "0.7")
        path = zeros(Int64, length(s1.aircraft_planned_dpath), 2)
        for t = 1:length(s1.aircraft_planned_dpath)
            path[t, 1], path[t, 2] = s1.aircraft_planned_dpath[t]
        end
        ax1[:plot](path[:, 2] - 1, path[:, 1] - 1, "--", color = "0.7")

        #ax1[:plot](s1.aircraft_path[1, 2] - 1, s1.aircraft_path[1, 1] - 1, "k.")
        ax1[:plot](s1.aircraft_start_loc[2] - 1, s1.aircraft_start_loc[1] - 1, "b^", markersize = 50 / params.n)

        ax1[:plot](params.uav_base_loc[2] - 1, params.uav_base_loc[1] - 1, "kx")

        if s1.uav_path != nothing
            #ax1[:plot](s1.uav_path[:, 2] - 1, s1.uav_path[:, 1] - 1, "r-.")
            #ax1[:plot](s1.uav_path[1, 2] - 1, s1.uav_path[1, 1] - 1, "k.")

            path = zeros(Int64, length(s1.uav_dpath), 2)
            for t = 1:length(s1.uav_dpath)
                path[t, 1], path[t, 2] = s1.uav_dpath[t]
            end
            ax1[:plot](path[:, 2] - 1, path[:, 1] - 1, "r-.")
        end

        ax1[:plot](params.uav_loc[2] - 1, params.uav_loc[1] - 1, "mo", markersize = 50 / params.n)

        #ax1[:text](0.5, -0.02, "", horizontalalignment = "center", verticalalignment = "top", transform = ax1[:transAxes])


        ax2 = fig[:add_subplot](222)
        ax2[:set_aspect]("equal")
        ax2[:invert_yaxis]()
        ax2[:set_xticks]([0.5:(params.n - 1.5)])
        ax2[:set_yticks]([0.5:(params.n - 1.5)])
        ax2[:set_xticklabels]([])
        ax2[:set_yticklabels]([])
        ax2[:grid](true)
        ax2[:set_title]("Utility")

        U_map = ax2[:imshow](U, cmap = "gray", alpha = 0.7, interpolation = "none", vmin = -200, vmax = 0)
        colorbar(U_map)

        #ax2[:text](0.5, -0.02, "", horizontalalignment = "center", verticalalignment = "top", transform = ax1[:transAxes])


        ax3 = fig[:add_subplot](223)
        ax3[:set_aspect]("equal")
        ax3[:invert_yaxis]()
        ax3[:set_xticks]([0.5:(params.n - 1.5)])
        ax3[:set_yticks]([0.5:(params.n - 1.5)])
        ax3[:set_xticklabels]([])
        ax3[:set_yticklabels]([])
        ax3[:grid](true)
        ax3[:set_title]("RE")

        RE_map = ax3[:imshow](RE, cmap = "gray_r", alpha = 0.7, interpolation = "none")
        colorbar(RE_map)

        #ax3[:text](0.5, -0.02, "", horizontalalignment = "center", verticalalignment = "top", transform = ax1[:transAxes])


        ax4 = fig[:add_subplot](224)
        ax4[:set_xlim](-0.5, 10.5)
        ax4[:set_ylim](-0.5, 10.5)
        ax4[:set_aspect]("equal")
        ax4[:invert_yaxis]()
        ax4[:set_xticks]([0.5:(params.n - 1.5)])
        ax4[:set_yticks]([0.5:(params.n - 1.5)])
        ax4[:set_xticklabels]([])
        ax4[:set_yticklabels]([])
        ax4[:grid](true)
        ax4[:set_title]("N")

        N_map = ax4[:imshow](N, cmap = "gray_r", alpha = 0.2, interpolation = "none")
        colorbar(N_map)

        for i = 1:params.n
            for j = 1:params.n
                ax4[:text](j - 1, i - 1, N[i, j], size = "5", horizontalalignment = "center", verticalalignment = "center")
            end
        end

        #ax4[:text](0.5, -0.02, "", horizontalalignment = "center", verticalalignment = "top", transform = ax1[:transAxes])
    end

    return U, RE, N, params
end


function plotPolicy(param_set_num::Int64, uncertainty::Float64; draw::Bool = true, fig = nothing, database::ASCIIString = "s1results.jld")

    U = Dict{Symbol, Any}()
    RE = Dict{Symbol, Any}()
    N = Dict{Symbol, Any}()
    params = Dict{Symbol, Any}()

    for policy in [:stay, :back, :landing]
        U[policy], RE[policy], N[policy], params[policy] = retrieveEvaluation(param_set_num, policy, uncertainty, database = database)
    end

    n = params[:back].n

    PM = zeros(Int64, n, n)

    for i = 1:n
        for j = 1:n
            U_tmp = Float64[]
            
            for policy in [:stay, :back, :landing]
                push!(U_tmp, U[policy][i, j])
            end

            ind = indmax(U_tmp)

            if ind == 2
                PM[i, j] = 1    # blue
            elseif ind == 1
                PM[i, j] = 2    # green
            elseif ind == 3
                PM[i, j] = 3    # red
            end
        end
    end

    if draw
        if fig == nothing
            fig = figure(facecolor = "white")
        end

        ax1 = fig[:add_subplot](111)
        ax1[:set_xlim](-0.5, 10.5)
        ax1[:set_ylim](-0.5, 10.5)
        ax1[:set_aspect]("equal")
        ax1[:invert_yaxis]()
        ax1[:set_xticks]([0.5:(n - 1.5)])
        ax1[:set_yticks]([0.5:(n - 1.5)])
        ax1[:set_xticklabels]([])
        ax1[:set_yticklabels]([])
        ax1[:grid](true)
        ax1[:set_title]("Policy")

        ax1[:imshow](PM, alpha = 0.5, interpolation = "none", vmin = 1, vmax = 3)

        params_ = params[:back]
        s1 = ScenarioOne(params_)
        initTrajectories(s1)

        #ax1[:plot](params_.wf_init_loc[2] - 1, params_.wf_init_loc[1] - 1, "rs", markersize = 100 / n)

        #ax1[:plot](s1.aircraft_path[:, 2] - 1, s1.aircraft_path[:, 1] - 1, "c--")

        #ax1[:plot](s1.aircraft_planned_path[:, 2] - 1, s1.aircraft_planned_path[:, 1] - 1, linestyle = "--", color = "0.7")
        path = zeros(Int64, length(s1.aircraft_planned_dpath), 2)
        for t = 1:length(s1.aircraft_planned_dpath)
            path[t, 1], path[t, 2] = s1.aircraft_planned_dpath[t]
        end
        ax1[:plot](path[:, 2] - 1, path[:, 1] - 1, "--", color = "0.7")

        #ax1[:plot](s1.aircraft_start_loc[2] - 1, s1.aircraft_start_loc[1] - 1, "b^", markersize = 100 / n)
        ax1[:plot](s1.aircraft_start_loc[2] - 1, s1.aircraft_start_loc[1] - 1, "k.")

        ax1[:plot](params_.uav_base_loc[2] - 1, params_.uav_base_loc[1] - 1, "kx")

        ax1[:text](0.5, -0.02, "red: emergency landing | green: stay in place | blue: back to base", horizontalalignment = "center", verticalalignment = "top", transform = ax1[:transAxes])
    end

    return PM, U, RE, N, params
end


function buildDatabase(database::ASCIIString; update::Bool = false, bParallel::Bool = false)

    for param_set in [1, 2]
        for policy in [:stay, :back, :landing]
            for uncertainty in [0., 1., 2.]
                retrieveEvaluation(param_set, policy, uncertainty, database = database, update = update, N_min = 1000, N_max = 10000, RE_threshold = 0.01, bParallel = bParallel)
            end
        end
    end
end

# $ julia -p 4 -L ScenarioOne_.jl -L simScenarioOne.jl ScenarioOneIJUtil.jl
#buildDatabase("s1results.jld", update = false, bParallel = true)


if false
    #retrieveEvaluation(1, :back, 0.)
    #plotEvaluation(1, :back, 1.)
    plotPolicy(1, 1.)
    readline()
end


