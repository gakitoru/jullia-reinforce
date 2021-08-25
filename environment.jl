using StatsBase

mutable struct State
    row
    column
end

function clone(model::State)
    return State(model.row, model.column)
end

@enum Action UP=1 DOWN=-1 LEFT=2 RIGHT=-2

mutable struct Environment
    grid
    agent_state
    default_reward
    move_prob
    function Environment(grid, agent_state=State(0, 0), default_reward=-0.04, move_prob=0.8)
        new(grid, agent_state, default_reward, move_prob)
    end
end

function row_length(env::Environment)
    size(env.grid)[1]
end

function column_length(env::Environment)
    size(env.grid[1])[1]
end

function actions(env::Environment)
    [UP::Action, DOWN::Action, LEFT::Action, RIGHT::Action]
end

function states(env::Environment)
    states = []
    for row = 1:row_length(env)
        for column = 1:column_length(env)
            if env.grid[row][column] != 9
                push!(states, State(row, column))
            end
        end
    end
    return states
end

function transit_func(env::Environment, state, action)
    transition_probs = Dict()

    if ! can_action_at(env, state)
        return transition_probs
    end

    if action == UP::Action
        opposite_direction = DOWN::Action
    elseif action == DOWN::Action
        opposite_direction = UP::Action
    elseif action == LEFT::Action
        opposite_direction = RIGHT::Action
    elseif action == RIGHT::Action
        opposite_direction = LEFT::Action
    end

    for a in actions(env)
        prob = 0
        if a == action
            prob = env.move_prob
        elseif a != opposite_direction
            prob = (1 - env.move_prob) / 2
        end

        next_state = _move(env, state, a)
        #println(transition_probs)
        if ! haskey(transition_probs, next_state)
            transition_probs[next_state] = prob
        else
            transition_probs[next_state] += prob
        end
    end
    return transition_probs
end

function can_action_at(env::Environment, state)
    if env.grid[state.row][state.column] == 0
        true
    else
        false
    end
end

function _move(env::Environment, state, action)
    if ! can_action_at(env, state)
        println("Can't move from here")
        throw(DomainError())
    end

    next_state = clone(state)

    # Execute an action (move).
    if action == UP::Action
        next_state.row -= 1
    elseif action == DOWN::Action
        next_state.row += 1
    elseif action == LEFT::Action
        next_state.column -= 1
    elseif action == RIGHT::Action
        next_state.column += 1
    end

    # Check whether a state if out of the grid
    if ! (1 <= next_state.row <= row_length(env))
        next_state = state
    end
    if ! (1 <= next_state.column <= column_length(env))
        next_state = state
    end
    if env.grid[next_state.row][next_state.column] == 9
        next_state = state
    end
    return next_state
end

function reward_func(env::Environment, state)
    reward = env.default_reward
    done = false

    # Check an attribute of next state.
    attribute = env.grid[state.row][state.column]
    if attribute == 1
        # Get reward! and the game ends.
        reward = 1
        done = true
    elseif attribute == -1
        # Go damage! and the game ends.
        reward = -1
        done = true
    end
    
    return reward, done 
end

function reset(env::Environment)
    # Locate the agent at lower left corner
    env.agent_state = State(row_length(env), 1)
    return env.agent_state
end

function step(env::Environment, action)
    next_state, reward, done = transit(env, env.agent_state, action)
    if next_state != nothing
        env.agent_state = next_state
    end
    return next_state, reward, done
end

function transit(env::Environment, state, action)
    transition_probs = transit_func(env, state, action)
    if length(transition_probs) == 0
        return nothing, nothing, true
    end

    next_states = []
    probs = []
    #println(transition_probs)
    for (s, v) in transition_probs
        #println(s)
        push!(next_states, s)
        push!(probs, v)
    end

    #probs = hcat(probs)
    probs = convert(Array{Float64, 1}, probs)
    next_state = sample(next_states, Weights(probs))
    reward, done = reward_func(env, next_state)
    return next_state, reward, done
end

# grid = [
#     [0, 0, 0, 1],
#     [0, 9, 0, -1],
#     [0, 0, 0, 0]
# ]

# env = Environment(grid)
# println(env)
# reset(env)

# a = actions(env)
# println(a[1])
# println(states(env))
# transit_func(env, State(1, 1), RIGHT::Action)
# print(reward_func(env, State(1, 4)))
# step(env, UP::Action)