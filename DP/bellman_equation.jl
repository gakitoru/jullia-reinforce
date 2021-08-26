function V(s, gamma=0.99)
    V = R(s) + gamma * max_V_on_next_state(s)
    return V
end

function R(s)
    if s == "happy_end"
        return 1
    elseif s == "bad_end"
        return -1
    else
        return 0
    end
end

function max_V_on_next_state(s)
    # If game end, expected value is 0.
    if s in ["happy_end", "bad_end"]
        return 0
    end

    actions = ["up", "down"]
    values = []
    for a in actions
        transition_probs = transit_func(s, a)
        v = 0
        for (next_state, prob) in transition_probs
            v += prob * V(next_state)
        end
        push!(values, v)
    end
    return maximum(values)
end

function transit_func(s, a)
    actions = split(s, "_")[2:end]
    LIMIT_GAME_COUNT = 5
    HAPPY_GAME_BORDER = 4
    MOVE_PROB = 0.9

    function next_state(state, action)
        return join([state, action], "_")
    end

    if length(actions) == LIMIT_GAME_COUNT
        up_count = sum([a == "up" ? 1 : 0 for a in actions])
        state = up_count >= HAPPY_GAME_BORDER ? "happy_end" : "bad_end"
        prob = 1.0
        return Dict(state => prob)
    else
        opposite = (a == "down" ? "up" : "down")
        return Dict(
            next_state(s, a) => MOVE_PROB,
            next_state(s, opposite) => 1 - MOVE_PROB
        )
    end

end

#println(max_V_on_next_state("happy"))
#transit_func("state_up_up_up", "up")

function main()
    println(V("state"))
    println(V("state_up_up"))
    println(V("state_down_down"))
end

main()