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
    states
end

env = Environment([[1, 2, 3, 4], [4, 5, 6, 7]], 1, 1, 1)
reset(env)

a = actions(env)
println(a[1])
println(states(env))