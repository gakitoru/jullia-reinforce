include("../environment.jl")
using ResumableFunctions

abstract type Planner 
end

mutable struct Planner_values
    env
    log
    function Planner_values(env)
        log = []
        new(env, log)
    end
end

function initialize(self::Planner)
    reset(self.planner.env)
    self.planner.log = []
end


function transitions_at(self::Planner, state, action)
    transition_probs = transit_func(self.planner.env, state, action)
    ret = []
    for (next_state, prob) in transition_probs
        reward, _ = reward_func(self.planner.env, next_state)
        push!(ret, (prob, next_state, reward))
    end
    return ret
end

function dict_to_grid(self::Planner, state_reward_dict)
    grid = []
    for i = 1:row_length(self.planner.env)
        row = zeros(column_length(self.planner.env))
        push!(grid, row)
    end
    
    for (s, r) in state_reward_dict
        grid[s.row][s.column] = r
    end
    return grid
end

mutable struct ValueIterationPlanner <: Planner
    planner::Planner_values
    function ValueIterationPlanner(env)
        new(Planner_values(env))
    end
end

# cannot use override
function plan(self::ValueIterationPlanner, gamma=0.9, threshold=0.0001)
    println("call plan")
    initialize(self)
    #actions = actions(self.planner.env)
    #Val = Dict{State, Float64}()
    Val = Dict()
    for s in states(self.planner.env)
        # initialize each state's expected reward.
        Val[s] = 0.0
        #println(s)
    end
    #println(Val)
    while 1==1
        delta = 0.0
        push!(self.planner.log, dict_to_grid(self, Val))
        for (s, _) in Val
            if ! can_action_at(self.planner.env, s)
                continue
            end
            expected_rewards = []
            for a in actions(self.planner.env)
                r = 0.0
                ret = transitions_at(self, s, a)
                probs = []
                next_states = []
                rewards = []
                for (prob, next_state, reward) in ret
                    push!(probs, prob)
                    push!(next_states, next_state)
                    push!(rewards, reward)
                end
                for (key, _) in Val
                    println("key:", key)
                    for (prob, next_state, reward) in zip(probs, next_states, rewards)
                        println("next_state:", next_state)
                        if key == next_state
                            r += prob * (reward + gamma * Val[key])
                            println("hit:", key)
                        end
                    end
                end
                println("")
                #println(r)
                push!(expected_rewards, r)
            end
            max_reward = maximum(expected_rewards)
            delta = delta > abs(max_reward - Val[s]) ? delta : abs(max_reward - Val[s])
            Val[s] = max_reward
        end
        if delta < threshold
            break
        end
    end
    V_grid = dict_to_grid(self, Val)
    return V_grid
end

grid = [
    [0, 0, 0, 1],
    [0, 9, 0, -1],
    [0, 0, 0, 0]
]

env = Environment(grid)
pl = ValueIterationPlanner(env)
ret = plan(pl)
println(ret)
println(typeof(ret))
A = [ret[x][y] for x in 1:1:3, y in 1:1:4]
println(A)
using Plots
heatmap(A)
savefig("out")

# for (prob, next_state, reward) in transitions_at(pl, State(1, 1), LEFT::Action)
#     println(prob)
#     println(next_state)
#     println(reward)
# end

