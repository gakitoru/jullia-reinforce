using Gym
using PyCall
using Statistics
using Plots

@pyimport gym.envs.registration as r
r.register(id="FrozenLakeEasy-v1", entry_point="gym.envs.toy_text:FrozenLakeEnv", kwargs=Dict("is_slippery"=>false))
#env = GymEnv("FrozenLakeEasy-v1")

function show_q_value(Q::Dict)
    env = GymEnv("FrozenLake-v1")
    nrow = env.gymenv.nrow
    ncol = env.gymenv.ncol
    state_size = 3
    q_nrow = nrow * state_size
    q_ncol = ncol * state_size
    reward_map = zeros((q_nrow, q_ncol))
    for r = 1:nrow
        for c = 1:ncol
            s = 1 + (r - 1) * nrow + (c - 1)
            #println(s)
            #state_exist = false
            #println(Q[1][1])
            #println(reward_map)
            #println(reward_map[7][3])
            _r = 2 + (nrow - r) * state_size
            _c = 2 + (c - 1) * state_size
            #println("reward_map($_r, $(_c))")
            reward_map[_r, _c - 1] = Q[s][1]
            reward_map[_r - 1, _c] = Q[s][2]
            reward_map[_r, _c + 1] = Q[s][3]
            reward_map[_r + 1, _c] = Q[s][4]
            reward_map[_r, _c] = mean(Q[s])
        end
    end
    heatmap(reward_map)
    savefig("out")
end

function show_q_value(Q::Matrix{Float64})
    env = GymEnv("FrozenLake-v1")
    nrow = env.gymenv.nrow
    ncol = env.gymenv.ncol
    state_size = 3
    q_nrow = nrow * state_size
    q_ncol = ncol * state_size
    reward_map = zeros((q_nrow, q_ncol))
    for r = 1:nrow
        for c = 1:ncol
            s = 1 + (r - 1) * nrow + (c - 1)
            #println(s)
            #state_exist = false
            #println(Q[1][1])
            #println(reward_map)
            #println(reward_map[7][3])
            _r = 2 + (nrow - r) * state_size
            _c = 2 + (c - 1) * state_size
            #println("reward_map($_r, $(_c))")
            reward_map[_r, _c - 1] = Q[s, 1]
            reward_map[_r - 1, _c] = Q[s, 2]
            reward_map[_r, _c + 1] = Q[s, 3]
            reward_map[_r + 1, _c] = Q[s, 4]
            reward_map[_r, _c] = mean(Q[s, :])
        end
    end
    heatmap(reward_map)
    savefig("out")
end

# Q = Dict()
# for i = 1:16
#     Q[i] = ones(4)
# end
# println(Q)
# show_q_value(Q)