import gym
from ray.rllib.env.policy_client import PolicyClient

def client_run_episode(
    env:gym.Env,
    client:PolicyClient,
    training_enabled:bool =True
):
    # reset environment and start episode
    obs = env.reset()
    eid = client.start_episode(training_enabled=training_enabled)

    done = False
    while not done:
        # get action from client
        action = client.get_action(eid, obs)
        # apply actions and observe returns
        obs, reward, done, info = env.step(action)
        # log returns
        if training_enabled:
            client.log_returns(eid, reward, info)

    # end episode
    client.end_episode(eid, obs)
