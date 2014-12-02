# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 11/30/2014

# Fast Informed Bound

module FIB_

export FIB, learn, selectAction


using Solver_
using POMDP_

using HDF5, JLD


import Solver_.selectAction


type FIB <: Solver

    policy_file::ASCIIString

    alpha::Union(Dict{Action, Dict{State, Float64}}, Nothing)

    gamma_::Float64
    eps::Float64


    function FIB(;model::Union(POMDP, Nothing) = nothing, policy_file::ASCIIString = "default.pcy", gamma_::Float64 = 0.9, eps::Float64 = 1.e-3)

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

    FIB(model::POMDP, policy_file::ASCIIString = "default.pcy"; gamma_::Float64 = 0.9, eps::Float64 = 1.e-3) = FIB(model = model, policy_file = policy_file, gamma_ = gamma_, eps = eps)

    FIB(policy_file::ASCIIString; gamma_::Float64 = 0.9, eps::Float64 = 1.e-3) = FIB(model = nothing, policy_file = policy_file, gamma_ = gamma_, eps = eps)
end


function learn(alg::FIB, pm::POMDP; policy_file::ASCIIString = "default.pcy")

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

        # O(|A|^2|S|^2|O|)
        for a in pm.actions
            # \alpha_a^{(k+1)}(s) = R(s,a) + \gamma \sum_o \max_{a'} \sum_{s'} O(o \mid s', a) T(s' \mid s, a) \alpha_{a'}^{(k)}(s')
            for s in pm.states
                sum_o = 0.

                for o in pm.observations
                    max_ = -Inf

                    for a_ in pm.actions
                        sum_ = 0.

                        for s_ in pm.states
                            sum_ += prob_obs(pm, s_, a, o) * prob_tran(pm, s, a, s_) * alpha[a_][s_]
                        end

                        if sum_ > max_
                            max_ = sum_
                        end
                    end

                    sum_o += max_
                end

                alpha_prev = alpha[a][s]
                alpha[a][s] = reward(pm, s, a) + alg.gamma_ * sum_o

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


function selectAction(alg::FIB, pm::POMDP, b::Belief)

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


