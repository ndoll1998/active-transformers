import gym
import torch
import numpy as np
# ignite
from ignite.engine import Events
from ignite.metrics import Metric
import ignite.distributed as idist
# import active learning components
from active.core.loop import ActiveLoop
from active.core.strategies import AbstractStrategy, Random
from active.helpers.engine import ActiveLearningEngine
from active.helpers.evaluator import Evaluator
# import data utilities
from active.core.utils.data import (
    NamedTensorDataset,
    default_collate_drop_labels
)
from torch.utils.data import (
    Subset, 
    Dataset,
    DataLoader,
    default_collate
)
# others
from itertools import chain
from functools import cached_property
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class State:
    """ Helper class storing the internal state of a `StreamBasedEnv` instance.

        Attributes:
            budget (int): total annotation budget
            query_size (int): number of queries to sample before retraining the model

            prev_metric (float): 
                value the reward metric evaluated to in the previous active learning step.
    """
    budget:int
    query_size:int

    # current query and corresponding observation
    # I.e. the query is an element extracted from the
    # model_queue and the observation is the corresponding
    # element in the observation queue
    query:dict =None
    obs:dict =None

    # model and observation queues
    model_queue:List[dict] = field(default_factory=list)    
    obs_queue:List[dict] =field(default_factory=list)

    # selected samples for current active learning step
    # and a counter to keep track of the total number of
    # samples selected
    samples:List[int] =field(default_factory=list)
    total_samples:int =0    

    # reward metric value evaluated at the previous AL step
    prev_metric:float =0

    def reset(self) -> None:
        self.__init__(
            budget=self.budget, 
            query_size=self.query_size
        )

    @property
    def are_queues_empty(self) -> bool:
        return len(self.obs_queue) == 0

    def next_query(self) -> dict:
        # make sure queues are not empty        
        assert not self.are_queues_empty, "Queues are empty!"
        # pop first element from both queues
        self.query = self.model_queue.pop()
        self.obs = self.obs_queue.pop()
        # return observation
        return self.obs

    def select_query(self) -> None:
        # check valid query
        assert self.query is not None, "No query set!"
        # add to samples
        self.samples.append(self.query)
        self.total_samples += 1
        # avoid selecting the query again
        self.query = None

    @property
    def query_index(self) -> int:
        return len(self.samples)

    @property
    def query_size_reached(self) -> bool:
        return self.query_index >= self.query_size

    @property
    def budget_exhausted(self) -> bool:
        return self.total_samples >= self.budget


class StreamBasedEnv(gym.Env):
    """ Stream-based Active Learning Environment for Natural Language Processing Tasks
        using Transformer Models and Policies.

        Stream-based refers to the setup that at each point the policy is presented
        with one sample only for which it has to decide whether to include it in the
        next active learning step or not.
    """

    def __init__(
        self,
        # active learning hyperparameters
        budget:int,
        query_size:int,
        # active learning engine
        engine:ActiveLearningEngine,
        # metric used to compute reward
        metric:Metric,
        # query selection strategy
        query_strategy:AbstractStrategy,
        # data
        policy_pool_data:Dataset,
        model_pool_data:Dataset,
        model_test_data:Dataset,
        # data setup, inferred from data
        # and model when not specified
        policy_sequence_length:Optional[int] =None,
        model_sequence_length:Optional[int] =None,
        max_num_labels:Optional[int] =None
    ) -> None:
        # initialize environment
        super(StreamBasedEnv, self).__init__()
        # check data pools
        assert len(policy_pool_data) == len(model_pool_data), "Size mismatch between data pools!"

        # create state instance
        self.state = State(
            budget=min(budget, len(policy_pool_data)),
            query_size=query_size
        )

        # save al engine and create placeholder variable
        # for evaluator which is instantiated in reset
        self.engine = engine
        self.evaluator:Evaluator = None
        # save reference to reward metric
        self.metric = metric
        # save reference to query selection strategy
        self.query_strategy = query_strategy

        # save references to datasets
        self.policy_pool_data = policy_pool_data
        self.model_pool_data = model_pool_data
        self.model_test_data = model_test_data

        # specify data layout, i.e. the dimensions of the observation space
        self._policy_seq_len = policy_sequence_length or len(self.policy_pool_data[0]['input_ids'])
        self._model_max_seq_len = model_sequence_length or self._model_seq_length
        self._model_max_num_labels = max_num_labels or self._model_num_labels

    @cached_property
    def _model_seq_length(self) -> int:
        return len(self.model_pool_data[0]['input_ids'])

    @cached_property
    def _model_num_labels(self) -> int:
        """ Actual number of labels of prediction model.
            Note that this may differ from the number of labels
            in the observation space to support different datasets
        """
        return self.engine.trainer.unwrapped_model.config.num_labels

    @property
    def _vocab_size(self) -> int:
        # instead of actually checking the vocab size lets just say the
        # vocab size is the maximum number that can be represented by int64
        # use int64 since it matches the torch.long datatype
        return np.iinfo(np.int64).max
        
    @property
    def _test_data_loader(self) -> DataLoader:
        return DataLoader(
            self.model_test_data, 
            batch_size=self.engine.eval_batch_size,
            shuffle=False
        )

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(2)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        # build observation space
        # TODO: support variable number of labels to apply to same model to different datasets
        #       by introducing a logits mask or something similar
        return gym.spaces.Dict({
            # policy transformer inputs
            "input_ids": gym.spaces.Box(
                low=0, 
                high=self._vocab_size, 
                shape=(self._policy_seq_len,), 
                dtype=np.int64
            ),
            "attention_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self._policy_seq_len,),
                dtype=np.bool
            ),
            # model predictions
            "logits": gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self._model_max_seq_len, self._model_max_num_labels),
                dtype=np.float32
            ),
            # model predictions mask
            "logits_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self._model_max_seq_len, self._model_max_num_labels),
                dtype=np.bool
            )
        })

    def _refill_queues(self) -> bool:
        # make sure queues are empty
        assert self.state.are_queues_empty, "Trying to refill non-empty queues!"
        
        # check if data pool is exhausted
        if len(self.loop.pool) == 0:
            return False

        # use initial sampling strategy as long as no active learning
        # step was done (here an activel learning step refers to training
        # the model)
        trained_once = self.engine.state.iteration > 0
        model_queue = self.loop.step() if trained_once else self.loop.init_step()

        # build policy queue corresponding to model queue    
        policy_queue = Subset(
            dataset=self.policy_pool_data, 
            indices=model_queue.indices
        )

        # get model predictions
        # note that by calling the evaluator's step function explicitly
        # no event are fired and thus the reward metric is not influenced
        batch = default_collate_drop_labels(model_queue)
        logits = self.evaluator.step(batch).logits.cpu()

        # expand logits to match expected number of labels in observation space
        expand = torch.zeros((logits.size(0), logits.size(1), self._model_max_num_labels - self._model_num_labels))
        logits = torch.cat((logits, expand), dim=2)
        # also expand to match expected sequence length
        expand = torch.zeros((logits.size(0), self._model_max_seq_len - logits.size(1), self._model_max_num_labels))
        logits = torch.cat((logits, expand), dim=1)

        # create logits mask
        logits_mask = torch.zeros((self._model_max_seq_len, self._model_max_num_labels), dtype=bool)
        logits_mask[:self._model_seq_length, :self._model_num_labels] = True

        # combine policy queue and model predictions
        # and write queues to state
        self.state.model_queue = list(model_queue)
        self.state.obs_queue = [
            {
                'input_ids': policy_item['input_ids'].numpy(),
                'attention_mask': policy_item['attention_mask'].numpy(),
                'logits': logits[i, ...].numpy(),
                'logits_mask': logits_mask.numpy()
            } for i, policy_item in enumerate(policy_queue)
        ]
        
        return True

    def reset(self):
        """ Reset the environment and return initial observation """

        # reset internal state
        self.state.reset()

        # call engine started event, this resets the
        # internals of the engine, i.e. it re-initializes
        # the model and resets the optimizer, scheduler
        self.engine._fire_event(Events.STARTED)      
        self.engine._fire_event(Events.EPOCH_STARTED)
        # for now the state has to be reset manually
        # TODO: there has to be a way of doing this in event handler
        self.engine._reset_state()

        # initialize evaluator and attach metric if not done before
        if self.evaluator is None:
            self.evaluator = Evaluator(self.engine.trainer.unwrapped_model)
            self.metric.attach(self.evaluator, "__reward_metric")

        # initialize active loop which is used to gather queue
        # elements which one-by-one are passed to the policy
        self.loop = ActiveLoop(
            pool=self.model_pool_data,
            batch_size=self.engine.train_batch_size,
            # each queue is of size eval_batch_size since
            # the whole queue needs to be passed through
            # the model to gather the predictions
            query_size=self.engine.eval_batch_size,
            # queue elements are sampled randomly
            # TODO: it might make sense to use a better strategy
            #       in later stages of the rl run to make the
            #       decision for the policy more complex
            strategy=self.query_strategy,
            init_strategy=Random()
        )

        # set initial queues
        self._refill_queues()

        # return query observation
        return self.state.next_query()

    def step(self, action):
        
        if bool(action):
            # select current query
            self.state.select_query()
        
        # info dictionary
        info = {}
        # get next observations and check
        # if done, i.e. budget is exhausted
        obs = self.state.next_query()
        done = self.state.budget_exhausted

        # check if queues are empty
        if self.state.are_queues_empty and (not done):
            # refill if so
            done |= not self._refill_queues()

        # train on selected samples
        if self.state.query_size_reached or (done and (self.state.query_index > 0)):
            # need to update the iteration manually as only the step
            # function (i.e. the process function) is called
            self.engine.state.iteration += 1
            # also call the iteration started event
            self.engine._fire_event(Events.ITERATION_STARTED)
            
            # create dataset from aquired data
            samples = default_collate(self.state.samples)
            samples = NamedTensorDataset(**samples)
            # clear samples in state
            self.state.samples.clear()

            # do an active learning step with the newly aquired data
            self.engine.step(samples)

            # evaluate model on test data and get reward metric
            state = self.evaluator.run(self._test_data_loader)
            metric_val = state.metrics['__reward_metric']
            # compute reward and update state
            reward = metric_val - self.state.prev_metric
            self.state.prev_metric = metric_val
            
            # call iteration completed event
            # note that at this point the evaluator has ran
            # and metrics depending on the evaluator can be computed
            self.engine._fire_event(Events.ITERATION_COMPLETED)

        else:
            # delayed reward
            reward = 0
        
        if done:
            # call engine completed event
            self.engine._fire_event(Events.EPOCH_COMPLETED)
            self.engine._fire_event(Events.COMPLETED)
            # add engine metrics to info
            info['custom_metrics'] = self.engine.state.metrics

        # return observation, reward, done and info
        return obs, reward, done, info

