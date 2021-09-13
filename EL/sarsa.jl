include("frozen_lake_util.jl")
include("el_agent.jl")
using DataStructures

mutable struct SARSAAgent <: ELAgent
    el_agent::ELAgent_value
    function SARSAAgent(epsilon::Float64=0.1)
        el_agent = ELAgent_value(epsilon)
        new(el_agent)
    end
end

function learn(self::SARSAAgent, env, episode_count=1000, gamma=0.9,
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
        a = policy(self, s, actions)
        a = a - 1
        while ! done
            if render
                env.gymenv.render()
            end
            n_state, reward, done, info = env.gymenv.step(a)
            a = a + 1
            n_state = n_state + 1
            n_action = policy(self, n_state, actions)
            gain = reward + gamma * self.el_agent.Q[n_state][n_action]
            estimated = self.el_agent.Q[s][a]
            self.el_agent.Q[s][a] += learning_rate * (gain - estimated)
            s = n_state
            n_action = n_action - 1
            a = n_action
        end
        if done
            log(self, reward)
        end
    end
end

function train()
    agent = SARSAAgent()
    env = GymEnv("FrozenLakeEasy-v1")
    learn(agent, env, 500)
    show_q_value(agent.el_agent.Q)
    show_reward_log(agent)
end

train()



