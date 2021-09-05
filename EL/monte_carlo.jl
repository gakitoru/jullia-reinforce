include("frozen_lake_util.jl")
#ENV["DISPLAY"]="xxx.xxx.xxx.xxx:0"
using DataStructures

mutable struct MonteCarloAgent <: ELAgent
    el_agent::ELAgent_value
    function MonteCarloAgent(epsilon=0.1)
        el_agent=ELAgent_value(epsilon)
        new(el_agent)
    end
end

function learn(self::MonteCarloAgent, env, episode_count=1000, gamma=0.9,
    render=false, report_interval=50)
    init_log(self.el_agent)
    actions = 1:env.action_space.n
    self.el_agent.Q = DefaultDict(zeros(length(actions)))
    N = DefaultDict(zeros(length(actions)))
    ## continue

    for e = 1:episode_count
        s = env.gymenv.reset()
        done = false
        # 1. Play until the end of episode.
        experience = []
        while ! done
            if render
                env.gymenv.render()
            end
            a = policy(self.el_agent, s, actions)
            n_state, reward, done, info = env.gymenv.step(a)
            push!(experience, Dict("state" => s, "action" => a, "reward" => reward))
            s = n_state
        else
            log(self.el_agent, reward)
        end
end
    
