include("frozen_lake_util.jl")
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
end
    
