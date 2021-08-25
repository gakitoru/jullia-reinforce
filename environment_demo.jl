include("environment.jl")
using StatsBase

mutable struct Agent
    actions
    function Agent(env)
        new(actions(env))
    end
end

function policy(agent::Agent, state)
    return sample(agent.actions)
end

function main()
    # Make grid environment.
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    for i = 1:10
        # Initialize position of agent.
        state = reset(env)
        total_reward = 0
        done = false

        while ! done
            action = policy(agent, state)
            next_state, reward, done = step(env, action)
            total_reward += reward
            state = next_state
        end
        println("Episode $i: Agent gets $total_reward reward.")
    end
end

main()