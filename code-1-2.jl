include("code-1-1.jl")

struct Environment
    grid
    agent_state
    default_reward
    move_prob
    function Environment(grid, agent_state, default_reward=-0.04, move_prob=0.8)
        new(grid, agent_state, default_reward, move_prob)
    end
end

function reset(model::Environment)
    println(model.grid)
end

function row_length(model::Environment)
    size(model.grid)[1]
end

function column_length(model::Environment)
    size(model.grid[1])[1]
end

function actions(model::Environment)
    [UP::Action, DOWN::Action, LEFT::Action, RIGHT::Action]
end

function states(model::Environment)
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

function transit_func(model::Environment, state, action)
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
        println(transition_probs)
        if ! haskey(transition_probs, next_state)
            transition_probs[next_state] = prob
        else
            transition_probs[next_state] += prob
        end
    end
    return transition_probs
end

function can_action_at(env, state)
    if env.grid[state.row][state.column] == 0
        true
    else
        false
    end
end

function _move(env, state, action)

    if ! can_action_at(env, state)
        println("Can't move from here")
    end

    next_state = clone(state)
    next_state
end

env = Environment([[0, 0, 0, 0], [0, 0, 0, 0]], 1, 1, 1)
reset(env)

a = actions(env)
println(a[1])
println(states(env))
transit_func(env, State(1, 1), RIGHT::Action)