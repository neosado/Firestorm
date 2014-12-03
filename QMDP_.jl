# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/30/2014

# Fully Observable Value Approximation

module QMDP_

export QMDP, learn, selectAction


using Solver_
using POMDP_

using HDF5, JLD


import Solver_.selectAction


type QMDP <: Solver

    policy_file::ASCIIString

    alpha::Union(Dict{Action, Dict{State, Float64}}, Nothing)

    gamma_::Float64
    eps::Float64


    function QMDP(;model::Union(POMDP, Nothing) = nothing, policy_file::ASCIIString = "default.pcy", gamma_::Float64 = 0.9, eps::Float64 = 1.e-3)

        self = new()

        if model != nothing
            self.gamma_ = gamma_
            self.eps = eps

            learn(self, model; policy_file = policy_file)
        end

        if isfile(policy_file)
            d = load(policy_file)

            self.alpha = d["alpha"]
            self.gamma_ = d["gamma"]
            self.eps = d["eps"]
        else
            self.alpha = nothing
            self.gamma_ = gamma_
            self.eps = eps
        end

        return self
    end

    QMDP(model::POMDP, policy_file::ASCIIString = "default.pcy"; gamma_::Float64 = 0.9, eps::Float64 = 1.e-3) = QMDP(model = model, policy_file = policy_file, gamma_ = gamma_, eps = eps)

    QMDP(policy_file::ASCIIString; gamma_::Float64 = 0.9, eps::Float64 = 1.e-3) = QMDP(model = nothing, policy_file = policy_file, gamma_ = gamma_, eps = eps)
end


function learn(alg::QMDP, pm::POMDP; policy_file::ASCIIString = "default.pcy")

    alpha = Dict{Action, Dict{State, Float64}}()
    for a = pm.actions
        alpha[a] = Dict{State, Float64}()

        for s in pm.states
            alpha[a][s] = 0.
        end
    end

    niter = 0

    while true
        res = 0.

        # O(|A|^2|S|^2)
        for a in pm.actions
            # \alpha_a^{(k+1)}(s) = R(s,a) + \gamma \sum_{s'} T(s' \mid s, a) \max_{a'} \alpha_{a'}^{(k)}(s').
            for s in pm.states
                sum_ = 0.

                for s_ in pm.states
                    max_ = -Inf

                    for a_ in pm.actions
                        u = alpha[a_][s_]
                        if u > max_
                            max_ = u
                        end
                    end

                    if pm.reward_functype == :type2
                        sum_ += tranProb(pm, s, a, s_) * max_
                    elseif pm.reward_functype == :type3
                        sum_ += tranProb(pm, s, a, s_) * (reward(pm, s, a, s_) + alg.gamma_ * max_)
                    end
                end

                alpha_prev = alpha[a][s]
                if pm.reward_functype == :type2
                    alpha[a][s] = reward(pm, s, a) + alg.gamma_ * sum_
                elseif pm.reward_functype == :type3
                    alpha[a][s] = sum_
                end

                res += (alpha[a][s] - alpha_prev)^2
            end
        end

        niter += 1

        if sqrt(res) < alg.eps
            break
        end
    end


    alg.alpha = alpha
    alg.policy_file = policy_file

    save(policy_file, "alpha", alpha, "niter", niter, "gamma", alg.gamma_, "eps", alg.eps)


    return alpha, niter
end


function selectAction(alg::QMDP, pm::POMDP, b::Belief)

    if alg.alpha == nothing
        error("alpha should be learned ahead.")
    end

    # a^* = \argmax_a \vect\alpha_a^\transpose \vect b

    U = Dict{Action, Float64}()
    U_max = -Inf

    for a in pm.actions
        U[a] = 0.

        for s in pm.states
            U[a] += alg.alpha[a][s] * b.belief[s]
        end

        if U[a] > U_max
            U_max = U[a]
        end
    end

    actions = Action[]
    for a in pm.actions
        if U[a] == U_max
            push!(actions, a)
        end
    end
    action = actions[rand(1:length(actions))]

    return action, U
end


end


