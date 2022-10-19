from ray.rllib.algorithms.callbacks import DefaultCallbacks

class CustomMetricsFromEnvCallbacks(DefaultCallbacks):
    """ Callbacks to forward custom metrics from Environemnt
        info dict to episode
    """

    def on_episode_step(self, episode, **kwargs):
        # add custom metrics from environment to episode
        episode.custom_metrics.update(
            episode.last_info_for().get("custom_metrics", {})
        )

class LoggingCallbacks(DefaultCallbacks):
    """ Logging callbacks """

    def on_episode_start(self, episode, env_index, **kwargs):
        print("Episode {} (env-idx={}) started".format(
            episode.episode_id, env_index
        ))

    def on_episode_end(self, episode, env_index, **kwargs):
        print("Episode {} (env-idx={}) ended after {} steps".format(
            episode.episode_id, env_index, episode.length
        ))
        print("Custom Metrics: {}".format(episode.custom_metrics))

    def on_sample_end(self, samples, **kwargs):
        if samples.count > 0:
            print("Returned sample batch of size {}".format(samples.count))

    def on_train_result(self, algorithm, result, **kwargs):
        print("Algorithm.train() result: {} -> {} episodes".format(
            algorithm, result['episodes_this_iter']
        ))

    def on_learn_on_batch(self, policy, **kwargs):
        print("policy.learn_on_batch() result: {}".format(policy))
