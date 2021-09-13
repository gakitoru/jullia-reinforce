include("frozen_lake_util.jl")
include("el_agent.jl")
#ENV["DISPLAY"]="xxx.xxx.xxx.xxx:0"
using DataStructures

mutable struct QLearningAgent <: ELAgent
    el_agent::ELAgent_value
    function QLearningAgent(epsilon::Float64 = 0.1)
        el_agent = ELAgent_value(epsilon)
        new(el_agent)
    end
end

function learn(self::QLearningAgent, env, episode_count=1000, gamma=0.9,
    learning_rate=0.1, render=false, report_interval=50)
    init_log(self)
    actions = 1:env.action_space.n
    self.el_agent.Q = Dict()
    for s = 1:env.observation_space.n
        self.el_agent.Q[s] = [0.0, 0.0, 0.0, 0.0]
    end

    reward = 0
    for e = 1:episode_count
        s = env.gymenv.reset()
        s = s + 1
        done = false
        while ! done
            if render
                env.gymenv.render()
            end
            a = policy(self, s, actions)
            a = a - 1
            n_state, reward, done, info = env.gymenv.step(a)
            a = a + 1
            n_state = n_state + 1
            gain = reward + gamma * maximum(self.el_agent.Q[n_state])
            estimated = self.el_agent.Q[s][a]
            self.el_agent.Q[s][a] += learning_rate * (gain - estimated)
            s = n_state
        end
        if done
            log(self, reward)
        end
    end
end

function train()
    agent = QLearningAgent(0.1)
    env = GymEnv("FrozenLakeEasy-v1")
    learn(agent, env, 500)
    show_q_value(agent.el_agent.Q)
    show_reward_log(agent)
end

train()

