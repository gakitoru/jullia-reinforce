include("frozen_lake_util.jl")
include("el_agent.jl")
using DataStructures

mutable struct Actor <: ELAgent
    el_agent::ELAgent_value
    actions
    Q
    function Actor(env)
        el_agent = ELAgent_value(-1.0)
        nrow = env.observation_space.n
        ncol = env.action_space.n
        new(el_agent, 1:env.action_space.n, rand(nrow, ncol))
    end
end

function softmax(self::Actor, x)
    x = map(y -> exp(y), x)
    return x / sum(x)
end

function policy(self::Actor, s)
    probs = softmax(self, self.Q[s, :])
    a = StatsBase.sample(self.actions, Weights(probs))
    return a
end

mutable struct Critic
    states
    V
    function Critic(env)
        states = env.observation_space.n
        new(states, zeros(states))
    end
end

mutable struct ActorCritic
    actor_class
    critic_class
    function ActorCritic(actor_class, critic_class)
        new(actor_class, critic_class)
    end
end

function train(self::ActorCritic, env, episode_count=1000, gamma=0.9,
    learning_rate=0.1, render=false, report_interval=50)
    actor = self.actor_class(env)
    critic = self.critic_class(env)

    init_log(actor)
    reward = 0
    for e = 1:episode_count
        s = env.gymenv.reset()
        s = s + 1
        done = false
        while ! done
            if render
                env.gymenv.render()
            end

            a = policy(actor, s)
            a = a - 1
            n_state, reward, done, info = env.gymenv.step(a)
            a = a + 1
            n_state = n_state + 1
            gain = reward + gamma * critic.V[n_state]
            estimated = critic.V[s]
            td = gain - estimated
            #println("s, a = $s, $a")
            #println(actor.Q)
            actor.Q[s, a] += learning_rate * td
            critic.V[s] += learning_rate * td
            s = n_state
        end
        if done
            log(actor, reward)
        end

        if e != 0 && e % report_interval == 0
            show_reward_log(actor, e)
        end
    end
    return actor, critic
end

function train()
    trainer = ActorCritic(Actor, Critic)
    env = GymEnv("FrozenLakeEasy-v1")
    actor, critic = train(trainer, env, 3000)
    show_q_value(actor.Q)
    show_reward_log(actor)
end

train()