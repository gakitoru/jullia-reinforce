using Random
using StatsBase
using Statistics
using Plots
#using DataFrameu

mutable struct Cointoss
    head_probs
    max_episode_steps
    toss_count
    function Cointoss(head_probs, max_episode_steps=30)
        toss_count = 0
        new(head_probs, max_episode_steps, toss_count)
    end
end

function __len__(self::Cointoss)
    return length(self.head_probs)
end

function reset(self::Cointoss)
    self.toss_count = 0
end

function step(self::Cointoss, action)
    final = self.max_episode_steps - 1
    if self.toss_count > final
        println("The step count exceeded maximum. Please reset env.")
        throw(DomainError())
    else
        done = self.toss_count == final ? true : false
    end

    if action > length(self.head_probs)
        println("The No.$action coin doesn't exist.")
        throw(DomainError())
    else
        head_prob = self.head_probs[action]
        if rand() < head_prob
            reward = 1.0
        else
            reward = 0.0
        end
        self.toss_count += 1
        return reward, done
    end
end

mutable struct EpsilonGreedyAgent
    epsilon
    V
    function EpsilonGreedyAgent(epsilon)
        V = []
        new(epsilon, V)
    end
end

function policy(self::EpsilonGreedyAgent)
    coins = 1:length(self.V)
    if rand() < self.epsilon
        return sample(coins)
    else
        return argmax(self.V)
    end
end

function play(self::EpsilonGreedyAgent, env)
    # initialize estimation
    N = zeros(__len__(env))
    self.V = zeros(__len__(env))
    reset(env)
    done = false
    rewards = []
    while ! done
        selected_coin = policy(self)
        reward, done = step(env, selected_coin)     
        push!(rewards, reward)
        n = N[selected_coin]
        coin_average = self.V[selected_coin]
        new_average = (coin_average * n + reward)  / (n + 1)
        N[selected_coin] += 1
        self.V[selected_coin] = new_average
    end
    return rewards
end

function main()
    env = Cointoss([0.1, 0.5, 0.1, 0.9, 0.1])
    epsilons = [0.0, 0.1, 0.2, 0.5, 0.8]
    game_steps = 10:10:310
    result = Dict()

    for e in epsilons
        agent = EpsilonGreedyAgent(e)
        means = []
        for s in game_steps
            env.max_episode_steps = s
            rewards = play(agent, env)
            push!(means, mean(rewards))
        end
        result["epsilon=$e"] = means
    end
    #result["coin toss count"] = game_steps
    println(result)
    for (k, v) in result
        plot!(v)
    end
    savefig("out")
end

main()