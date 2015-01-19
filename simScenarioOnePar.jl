# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/19/2015

# $ julia -p 4 -L ScenarioOne_.jl -L simScenarioOne.jl simScenarioOnePar.jl

require("simScenarioOne")
using ScenarioOne_


U, RE, N, params = evaluatePolicy(1, :back, 0., N_min = 1000, N_max = 10000, RE_threshold = 0.01, bParallel = true)

# debug
println(params)
println(int(U))
println(round(RE, 4))
println(N)


