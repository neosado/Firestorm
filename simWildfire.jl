# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 05/05/2015

using Wildfire_
using WildfireVisualizer_


function simulate_wildfire(wm, ts_sim = 1; plots = :all, draw = false, wait = false, bLabel = true)

    if draw
        wfv = WildfireVisualizer(plots = plots, wait = wait, bLabel = bLabel)

        visInit(wfv, wm)
        visUpdate(wfv, wm)
        updateAnimation(wfv)
    end

    for t = 1:ts_sim
        wfNextState(wm, bFuel = true)

        if draw
            visInit(wfv, wm)
            visUpdate(wfv, wm, t)
            updateAnimation(wfv)
        end
    end

    if draw
        saveAnimation(wfv)
        readline()
    end
end


nrow = 11
ncol = 11
#wf_seed = uint64(time())
#wf_seed = uint64(rand(1:1024))
wf_seed = uint64(293)
wf_init_loc = [(6, 4), (7, 4), (5, 5), (6, 5), (7, 5), (4, 6), (5, 6), (6, 6), (7, 6), (5, 7), (6, 7), (6, 8)]
wf_p_fire = 0.06
wf_sim_time = 60
wait = true

wm = Wildfire(nrow, ncol, seed = wf_seed, init_loc = wf_init_loc, p_fire = wf_p_fire)

simulate_wildfire(wm, wf_sim_time, plots = :all, draw = true, wait = wait, bLabel = true)


