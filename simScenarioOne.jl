# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 01/13/2015

using Wildfire_
using WildfireVisualizer_

using ScenarioOne_
using ScenarioOneVisualizer_

using PyPlot
using Base.Test


function simulate(;draw = false, wait = false)

    s1 = ScenarioOne(11, p_fire = 0.12)

    state = ScenarioOneState(s1)

    initTrajectories(s1)


    if draw
        s1v = ScenarioOneVisualizer(wait = wait)

        visInit(s1v, s1)
        visUpdate(s1v, s1)
        updateAnimation(s1v)
    end

    t = 0
    U = 0

    while !isEndState(s1, state)
        updateLocations(s1, state, t)

        r = getReward(s1, state)
        U += r

        if draw
            visInit(s1v, s1)
            visUpdate(s1v, s1, t, state, r, U)
            updateAnimation(s1v)
        end

        updateStatus(s1, state)

        t += 1
    end

    if draw
        saveAnimation(s1v, repeat = true)
    end
end


simulate(draw = true, wait = true)


