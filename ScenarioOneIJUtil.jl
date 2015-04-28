# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/13/2015

global __PARALLEL__
if !isdefined(:__PARALLEL__)
    __PARALLEL__ = false
end

if !__PARALLEL__
using PyPlot
end

using HDF5, JLD


include("simScenarioOne.jl")


function retrieveEvaluation(version::ASCIIString, param_set_num::Int64, policy::Symbol; datafile::ASCIIString = "s1results.jld", update::Bool = false, sim_comm_loss_duration_mu::Union(Float64, Nothing) = nothing, sim_comm_loss_duration_sigma::Float64 = 0., sim_continue::Bool = false, r_surveillance::Float64 = 0., uav_surveillance_pattern::Union(Symbol, Nothing) = nothing, aircraft_traj_uncertainty::Union(Float64, Nothing) = nothing, N_min::Int = 100, N_max::Int = 1000, RE_threshold::Float64 = 0.01, bParallel::Bool = false)

    if isfile(datafile)
        database = load(datafile, "DATA")
    else
        database = Dict{Uint64, Any}()
    end

    if version == "0.1"
        @assert aircraft_traj_uncertainty != nothing
        key = hash([param_set_num, policy, aircraft_traj_uncertainty])

    elseif version == "0.2" || version == "0.2.1"
        @assert sim_comm_loss_duration_mu != nothing
        key = hash([param_set_num, policy, sim_comm_loss_duration_mu, sim_comm_loss_duration_sigma])

    elseif version == "0.3"
        @assert sim_continue
        key = hash([param_set_num, policy, r_surveillance])

    elseif version == "1.0"
        @assert uav_surveillance_pattern != nothing
        key = hash([param_set_num, policy, uav_surveillance_pattern])

    end

    if update != true && haskey(database, key)
        params = database[key]["params"]
        U = database[key]["U"]
        RE = database[key]["RE"]
        N = database[key]["N"]

    else
        U, RE, N, params = evaluatePolicy(version, param_set_num, policy, sim_comm_loss_duration_mu = sim_comm_loss_duration_mu, sim_comm_loss_duration_sigma = sim_comm_loss_duration_sigma, sim_continue = sim_continue, r_surveillance = r_surveillance, uav_surveillance_pattern = uav_surveillance_pattern, aircraft_traj_uncertainty = aircraft_traj_uncertainty, N_min = N_min, N_max = N_max, RE_threshold = RE_threshold, bParallel = bParallel)
        
        record = Dict{ASCIIString, Any}()
        record["params"] = params
        record["U"] = U
        record["RE"] = RE
        record["N"] = N
        database[key] = record

        save(datafile, "DATA", database)
    end
    
    # debug
    #println(params)
    #println(int(U))
    #println(round(RE, 4))
    #println(N)
    
    return U, RE, N, params
end


function plotEvaluation(version::ASCIIString, param_set_num::Int64, policy::Symbol; draw::Bool = true, fig = nothing, datafile::ASCIIString = "s1results.jld", sim_comm_loss_duration_mu::Union(Float64, Nothing) = nothing, sim_comm_loss_duration_sigma::Float64 = 0., sim_continue::Bool = false, r_surveillance::Float64 = 0., uav_surveillance_pattern::Union(Symbol, Nothing) = nothing, aircraft_traj_uncertainty::Union(Float64, Nothing) = nothing)
    
    U, RE, N, params = retrieveEvaluation(version, param_set_num, policy, datafile = datafile, sim_comm_loss_duration_mu = sim_comm_loss_duration_mu, sim_comm_loss_duration_sigma = sim_comm_loss_duration_sigma, sim_continue = sim_continue, r_surveillance = r_surveillance, uav_surveillance_pattern = uav_surveillance_pattern, aircraft_traj_uncertainty = aircraft_traj_uncertainty)

    # debug
    #println(map(x -> round(signif(x, 4), 4), U))
    #println(map(x -> round(signif(x, 4), 4), RE))

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
        if isa(params.wf_init_loc, (Int64, Int64))
            B[params.wf_init_loc[1], params.wf_init_loc[2]] = true
        else
            for (row, col) in params.wf_init_loc
                B[row, col] = true
            end
        end
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

        if sim_continue
            vmax_ = 10
        else
            vmax_ = 0
        end
        U_map = ax2[:imshow](U, cmap = "gray", alpha = 0.7, interpolation = "none", vmin = -200, vmax = vmax_)
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


function plotPolicy(version::ASCIIString, param_set_num::Int64; draw::Bool = true, fig = nothing, datafile::ASCIIString = "s1results.jld", sim_comm_loss_duration_mu::Union(Float64, Nothing) = nothing, sim_comm_loss_duration_sigma::Float64 = 0., sim_continue::Bool = false, r_surveillance::Float64 = 0., uav_surveillance_pattern::Union(Symbol, Nothing) = nothing, T::Float64 = 1., aircraft_traj_uncertainty::Union(Float64, Nothing) = nothing)

    if version == "0.1"
        policies = [:stay, :back, :landing]
    else
        policies = [:stay, :back, :landing, :lower]
    end

    U = Dict{Symbol, Any}()
    RE = Dict{Symbol, Any}()
    N = Dict{Symbol, Any}()
    params = Dict{Symbol, Any}()

    if version == "1.0" && uav_surveillance_pattern == :mixed
        for policy in policies
            patterns = [:mower, :chase, :back]
            npatterns = length(patterns)

            U_ = Array(Any, npatterns)
            RE_ = Array(Any, npatterns)
            N_ = Array(Any, npatterns)

            for i = 1:npatterns
                U_[i], RE_[i], N_[i], params_ = retrieveEvaluation(version, param_set_num, policy, datafile = datafile, sim_comm_loss_duration_mu = sim_comm_loss_duration_mu, sim_comm_loss_duration_sigma = sim_comm_loss_duration_sigma, sim_continue = sim_continue, r_surveillance = r_surveillance, uav_surveillance_pattern = patterns[i], aircraft_traj_uncertainty = aircraft_traj_uncertainty)
            end

            U[policy] = zeros(params_.n, params_.n)
            RE[policy] = zeros(params_.n, params_.n)
            N[policy] = zeros(Int64, params_.n, params_.n)

            P = Array(Any, npatterns)
            for i = 1:npatterns
                P[i] = exp(U_[i] / T)
            end
            P_sum = sum(P)

            for i = 1:npatterns
                U[policy] += P[i] ./ P_sum .* U_[i]
            end

            for i = 1:params_.n
                for j = 1:params_.n
                    RE__ = zeros(npatterns)
                    N__ = zeros(npatterns)
                    for k = 1:npatterns
                        RE__[k] = RE_[k][i, j]
                        N__[k] = N_[k][i, j]
                    end
                    RE[policy][i, j] = maximum(RE__)
                    N[policy][i, j] = maximum(N__)
                end
            end
            params[policy] = params_

            # debug
            #row = 10
            #col = 6
            #println("==== ", policy, " ====")
            #println(U_[1][row, col])
            #println(U_[2][row, col])
            #println(U_[3][row, col])
            #println(U[policy][row, col])
        end
    else
        for policy in policies
            U[policy], RE[policy], N[policy], params[policy] = retrieveEvaluation(version, param_set_num, policy, datafile = datafile, sim_comm_loss_duration_mu = sim_comm_loss_duration_mu, sim_comm_loss_duration_sigma = sim_comm_loss_duration_sigma, sim_continue = sim_continue, r_surveillance = r_surveillance, uav_surveillance_pattern = uav_surveillance_pattern, aircraft_traj_uncertainty = aircraft_traj_uncertainty)
        end
    end

    n = params[:back].n

    PM = zeros(Int64, n, n)

    for i = 1:n
        for j = 1:n
            U_tmp = Float64[]
            
            for policy in policies
                push!(U_tmp, U[policy][i, j])
            end

            ind = indmax(U_tmp)

            if length(policies) == 3
                # :stay, :back, :landing
                if ind == 2
                    PM[i, j] = 1    # blue
                elseif ind == 1
                    PM[i, j] = 2    # green
                elseif ind == 3
                    PM[i, j] = 3    # red
                end
            elseif length(policies) == 4
                # :stay, :back, :landing, :lower
                if ind == 2
                    PM[i, j] = 1    # blue
                elseif ind == 1
                    PM[i, j] = 2    # cyan
                elseif ind == 4
                    PM[i, j] = 3    # yellow
                elseif ind == 3
                    PM[i, j] = 4    # red
                end
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

        ax1[:imshow](PM, alpha = 0.5, interpolation = "none", vmin = 1, vmax = length(policies))

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

        if length(policies) == 3
            ax1[:text](0.5, -0.02, "red: emergency landing | green: stay in place | blue: back to base", horizontalalignment = "center", verticalalignment = "top", transform = ax1[:transAxes])
        elseif length(policies) == 4
            #ax1[:text](0.5, -0.02, "red: emergency landing | cyan: stay in place | blue: back to base | yellow: lower altitude", color = "red", horizontalalignment = "center", verticalalignment = "top", transform = ax1[:transAxes])
            ax1[:text](0.01, -0.02, "emergency landing", color = "#B87575", horizontalalignment = "left", verticalalignment = "top", transform = ax1[:transAxes])
            ax1[:text](0.36, -0.02, "stay in place", color = "#75E7FF", horizontalalignment = "left", verticalalignment = "top", transform = ax1[:transAxes])
            ax1[:text](0.60, -0.02, "back to base", color = "#7575B8", horizontalalignment = "left", verticalalignment = "top", transform = ax1[:transAxes])
            ax1[:text](0.84, -0.02, "lower altitude", color = "#FFF175", horizontalalignment = "left", verticalalignment = "top", transform = ax1[:transAxes])
        end
    end

    return PM, U, RE, N, params
end


if false
    #retrieveEvaluation("0.1", 1, :back, datafile = "s1results_v0_1.jld", aircraft_traj_uncertainty = 0.)
    #plotEvaluation("0.1", 1, :back, aircraft_traj_uncertainty = 1.)
    plotPolicy("0.1", 1, aircraft_traj_uncertainty = 1.)
    readline()
end


