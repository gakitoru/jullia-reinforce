include("../environment.jl")
using ResumableFunctions
using Plots

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

function initialize(self::Planner_values)
    reset(self.env)
    self.log = []
end

function transitions_at(self::Planner, state, action)
    transition_probs = transit_func(self.planner.env, state, action)
    ret = []
    for next_state in keys(transition_probs)
        prob = transition_probs[next_state]
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
    initialize(self.planner)
    Val = Dict{State, Float64}()
    for s in states(self.planner.env)
        #initialize each state's expected reward.
        Val[s] = 0.0
    end
    while true
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
                for (prob, next_state, reward) in ret
                    for key in keys(Val)
                        if key.column == next_state.column && key.row == next_state.row
                            r += prob * (reward + gamma * Val[key])
                        end
                    end
                end
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

mutable struct PolicyIterationPlanner <: Planner
    planner::Planner_values
    policy
    function PolicyIterationPlanner(env)
        policy = Dict()
        new(Planner_values(env), policy)
    end
end

function initialize(self::PolicyIterationPlanner)
    initialize(self.planner)
    self.policy = Dict()
    #println(self.planner.env)
    acts = actions(self.planner.env)
    stats = states(self.planner.env)
    for s in stats
        self.policy[s] = Dict()
        for a in acts
            self.policy[s][a] = 1 / length(acts)
        end
    end
    #println(self.policy)
end

function estimate_by_policy(self::PolicyIterationPlanner, gamma, threshold)
    V = Dict()
    for s in states(self.planner.env)
        V[s] = 0
    end

    while true
        delta = 0
        for s in keys(V)
            expected_rewards = []
            #println(self.policy)
            pol_key = nothing
            for k in keys(self.policy)
                if k.column == s.column && k.row == s.row
                    pol_key = k
                end
            end
            for a in keys(self.policy[pol_key])
                action_prob = self.policy[pol_key][a]
                r = 0
                for (prob, next_state, reward) in transitions_at(self, s, a)
                    for key in keys(V)
                        if key.column == next_state.column && key.row == next_state.row
                            r += action_prob * prob * (reward + gamma * V[key])
                        end
                    end
                end
                push!(expected_rewards, r)
            end
            max_reward = maximum(expected_rewards)
            delta = delta > abs(max_reward - V[s]) ? delta : abs(max_reward - V[s])
            V[s] = max_reward
        end
        if delta < threshold
            break
        end
    end
    return V
end

function plan(self::PolicyIterationPlanner, gamma=0.9, threshold=0.0001)
    initialize(self)
    stats = states(self.planner.env)
    acts = actions(self.planner.env)
    V = Dict()
    function take_max_action(action_value_dict)
        return reduce((x, y) -> action_value_dict[x] >= action_value_dict[y] ? x : y, keys(action_value_dict))
    end
    while true
        update_stable = true
        # Estimate expected rewards under current policy
        V = estimate_by_policy(self, gamma, threshold)
        push!(self.planner.log, V)

        for s in stats
            # Get an action following to the current policy
            pol_key = nothing
            for k in keys(self.policy)
                if k.column == s.column && k.row == s.row
                    pol_key = k
                end
            end
            policy_action = take_max_action(self.policy[pol_key])

            # compare with other actions
            action_rewards = Dict()
            for a in acts
                r = 0
                for (prob, next_state, reward) in transitions_at(self, s, a)
                    for key in keys(V)
                        if key.column == next_state.column && key.row == next_state.row
                            r += prob * (reward + gamma * V[key])
                        end
                    end
                end
                action_rewards[a] = r
            end
            best_action = take_max_action(action_rewards)
            if policy_action != best_action
                update_stable = false
            end

            # Update policy (set best_action prob=1, otherwise=0 (greedy)).
            for a in keys(self.policy[pol_key])
                prob = a == best_action ? 1 : 0
                self.policy[pol_key][a] = prob
            end
        end
        if update_stable
            break
        end
    end

    V_grid = dict_to_grid(self, V)
    return V_grid
end



grid = [
    [0, 0, 0, 1],
    [0, 9, 0, -1],
    [0, 0, 0, 0]
]

env = Environment(grid)
# pl = ValueIterationPlanner(env)
# ret = plan(pl)
# A = [ret[x][y] for x in 1:1:3, y in 1:1:4]
# heatmap(A)
# savefig("out")
pl = PolicyIterationPlanner(env)
ret = plan(pl)
A = [ret[x][y] for x in 1:1:3, y in 1:1:4]
heatmap(A)
savefig("out")


# for (prob, next_state, reward) in transitions_at(pl, State(1, 1), LEFT::Action)
#     println(prob)
#     println(next_state)
#     println(reward)
# end

