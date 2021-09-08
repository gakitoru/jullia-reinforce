include("frozen_lake_util.jl")
include("el_agent.jl")
#ENV["DISPLAY"]="xxx.xxx.xxx.xxx:0"
using DataStructures

mutable struct MonteCarloAgent <: ELAgent
    el_agent::ELAgent_value
    function MonteCarloAgent(epsilon::Float64)
        el_agent=ELAgent_value(epsilon)
        new(el_agent)
    end
end

function learn(self::MonteCarloAgent, env, episode_count=10000, gamma=0.9,
    render=false, report_interval=50)
    init_log(self)
    actions = 1:env.action_space.n
    #self.el_agent.Q = DefaultDict(zeros(length(actions)))
    #N = DefaultDict(zeros(length(actions)))
    self.el_agent.Q = Dict()
    N = Dict()
    for s = 1:env.observation_space.n
        self.el_agent.Q[s] = [0.0, 0.0, 0.0, 0.0]
        N[s] = [0.0, 0.0, 0.0, 0.0]
    end
    ## continue
    reward = 0
    for e = 1:episode_count
        s = env.gymenv.reset()
        s = s + 1
        done = false
        # 1. Play until the end of episode.
        experience = []
        while ! done
            if render
                env.gymenv.render()
            end
            a = policy(self, s, actions)
            a = a - 1
            n_state, reward, done, info = env.gymenv.step(a)
            if reward > 0
                println("======================== reward: $reward =================================")
            end
            a = a + 1
            push!(experience, Dict("state" => s, "action" => a, "reward" => reward))
            s = n_state + 1
        end
        if done
            log(self, reward)
        end

        println(length(experience))
        # 2. Evaluate each state, action.
        for (i, x) in enumerate(experience)
            s, a = x["state"], x["action"]

            # Calculate discounted future reward of s.
            G, t = 0, 0
            for j = i:length(experience)
                rew = experience[j]["reward"]
                #println("gamma: $gamma, t: $t, experience[$j]['reward']: $rew")
                G += (gamma^t) * experience[j]["reward"]
                t += 1
            end

            N[s][a] += 1.0
            alpha = 1.0 / N[s][a]
            self.el_agent.Q[s][a] += alpha * (G - self.el_agent.Q[s][a])
            println("N[$s][$a]: $(N[s][a])")
            println("G: $G")
            println("self.el_agent.Q[$s][$a]: $(self.el_agent.Q[s][a])")
            println("alpha: $alpha")
            println("")
        end
        println("==============================")

        if e != 0 && e % report_interval == 0
            show_reward_log(self, e)
        end
    end

end
    
function train()
    agent = MonteCarloAgent(0.1)
    env = GymEnv("FrozenLakeEasy-v1")
    learn(agent, env, 500)
    show_q_value(agent.el_agent.Q)
    show_reward_log(agent)
end

train()
    