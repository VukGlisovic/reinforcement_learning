import gym
import numpy as np
import cv2
from collections import deque
from tf_agents.environments.tf_py_environment import TFPyEnvironment


class NoopResetEnv(gym.Wrapper):
    """By executing the NOOP action a random number of steps at the
    start of an episode, we basically sample initial states and make
    sure we explore better.
    """

    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.max_num_noops = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            num_noops = self.override_num_noops
        else:
            num_noops = np.random.randint(1, self.max_num_noops + 1)
        obs = None
        for _ in range(num_noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class SkipFramesEnv(gym.Wrapper):
    """Skip configured amount of frames and repeat the action
    'skip' amount of times. At the same time, it takes the max
    pixel values over the last two frames.
    """

    def __init__(self, env, skip=4):
        super(SkipFramesEnv, self).__init__(env)
        # most recent raw observations for max pooling across time steps
        self.obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self.skip = skip

    def step(self, action):
        """Repeat the taken action, sum up the rewards and take the
        max pixel values over last two observations.
        """
        total_reward = 0.0
        done = None
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            if i == self.skip - 2:
                self.obs_buffer[0] = obs
            if i == self.skip - 1:
                self.obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation frame when done=True doesn't matter
        max_frame = self.obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class TimeLimitEnv(gym.Wrapper):

    def __init__(self, env, max_episode_steps=None):
        super(TimeLimitEnv, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class FireResetEnv(gym.Wrapper):
    """Take the 'FIRE' action on reset which is useful for environments
    that are fixed until firing.
    """

    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)  # action=1 means 'FIRE'
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life the same as end-of-episode, but only reset on a
    true game-over. This can be useful for value estimation according to
    DeepMind.
    """

    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        # below attributes will be set to the correct values after the first step
        self.lives = 0
        self.game_over = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.game_over = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives:
            done = True  # if game_over=True, then done was already True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.game_over:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class ClipRewardEnv(gym.RewardWrapper):
    """Bin rewards to {+1, 0, -1} by its sign.
    """

    def __init__(self, env):
        super(ClipRewardEnv, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class WarpFrameEnv(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.
    If in addition grayscale=True is configured, then each RGB frame will
    be converted to a grayscale image of size 84x84x1.

    If the environment uses dictionary observations, `dict_space_key` can
    be specified which indicates which observation should be warped.
    """

    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        super(WarpFrameEnv, self).__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStackEnv(gym.Wrapper):
    """Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """

    def __init__(self, env, k):
        super(FrameStackEnv, self).__init__(env)
        self.k = k
        # with deque: when (k+1)th element is appended, then the first element will automatically be removed
        self.frames = deque([], maxlen=k)
        shape_in = env.observation_space.shape
        shape_out = (shape_in[:-1] + (shape_in[-1] * k,))  # stacking k frames
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape_out, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrameEnv(gym.ObservationWrapper):
    """Scale the frame to interval [0, 1].
    """

    def __init__(self, env):
        super(ScaledFloatFrameEnv, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    """This object ensures that common frames between the observations
    are only stored once. It exists purely to optimize memory usage
    which can be huge for DQN's 1M frames replay buffers. This object
    should only be converted to numpy array before being passed to the
    model. You'd not believe how complex the previous solution was.
    """

    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


def wrap_atari_deepmind(env, frame_skip=4, episode_life=True, clip_rewards=True, frame_stack=False, scale=False, steps_limit=None):
    """Configure environment for DeepMind-style Atari.
    """
    env = NoopResetEnv(env, noop_max=30)
    if frame_skip:
        env = SkipFramesEnv(env, frame_skip)
    if steps_limit:
        env = TimeLimitEnv(env, steps_limit)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrameEnv(env)
    if scale:
        env = ScaledFloatFrameEnv(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStackEnv(env, 4)
    return env


def hard_reset(environment):
    """Due to EpisodicLifeEnv, the reset only works as expected if the
    attribute 'game_over' is True. This helper function makes sure this
    is the case and resets the environment to the very start.
    """

    def get_env(env):
        return env.env

    if isinstance(environment, TFPyEnvironment):
        e = environment.pyenv.envs[0]
    else:
        e = environment
    while not isinstance(e, EpisodicLifeEnv):
        e = get_env(e)
    e.game_over = True

    return environment.reset()


def get_timelimit_env(environment):
    """Helper function to get the TimeLimitEnv so that number of
    elapsed steps can be obtained for example.
    """

    def get_env(env):
        return env.env

    if isinstance(environment, TFPyEnvironment):
        e = environment.pyenv.envs[0]
    else:
        e = environment
    while not isinstance(e, TimeLimitEnv):
        e = get_env(e)

    return e
