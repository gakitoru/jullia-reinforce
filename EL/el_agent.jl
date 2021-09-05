using Plots
using Statistics
using StatsBase

mutable struct ELAgent
    Q
    epsilon
    reward_log
    function ELAgent(epsilon)
        Q = Dict()
        reward_log = []
        new(Q, epsilon, reward_log)
    end
end

function policy(self::ELAgent, s, actions)
    if rand() < self.epsilon
        return sample(1:length(actions))
    else
        if s in keys(self.Q) && sum(self.Q[s]) != 0
            return argmax(Q[s])
        else
            return sample(1:length(actions))
        end
    end
end

function init_log(self::ELAgent)
    self.reward_log = []
end

function log(self::ELAgent, reward)
    push!(self.reward_log, reward)
end

function show_reward_log(self::ELAgent, interval=50, episode=-1)
    if episode > 0
        rewards = self.reward_log[end-interval:end]
        mean = round(mean(rewards), digits=3)
        std = round(std(rewards), digits=3)
        println("At Episode $episode average reward is $mean (+/-$std)")
    else
        indices = 1:interval:length(self.reward_log)
        means = []
        stds = []
        for i in indices
            rewards = self.reward_log[i:(i + interval)]
            push!(means, mean(rewards))
            push!(stds, std(rewards))
        end

        plot!(indices, means)
        plot!(indices, means - stds)
        plot!(indices, means + stds)
        savefig("rewards")
    end
end
