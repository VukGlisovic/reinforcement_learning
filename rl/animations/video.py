import os
import io
import shutil
import base64
from gym import wrappers
from IPython.display import display, HTML


def write_video(env, action_fnc, max_steps=1000, write_dir='video', remove_old_video=True):
    """Writes a video to disk.

    Args:
        env (gym.env):
        action_fnc (callable): function accepting the environment observation
            and returning an action.
        max_steps (int): max steps of episode
        write_dir (str): where to store the video
        remove_old_video (bool): whether to remove the old video

    Returns:
        tuple[str, float, int]
    """
    if remove_old_video:
        shutil.rmtree('video', ignore_errors=True)
    env_wrapper = wrappers.Monitor(env, write_dir)

    step = 0
    total_reward = 0
    terminal = False

    observation = env_wrapper.reset()
    env_wrapper.render()
    while not terminal and step < max_steps:
        action = action_fnc(observation)
        observation, reward, terminal, info = env_wrapper.step(action)
        step += 1
        total_reward += reward
        env_wrapper.render()
    env_wrapper.close()

    filename_mp4 = [fn for fn in os.listdir(write_dir) if fn.endswith('.mp4')][0]
    path_mp4 = os.path.join(write_dir, filename_mp4)

    return path_mp4, total_reward, step


def show_video(path, width=800, play_type='controls'):
    """Plays a generated video.

    Args:
        path (str): path to the video file
        width (int): width of the video. The height will automatically
            be adjusted to the width.
        play_type (str): 'autoplay' or 'controls'
    """
    assert os.path.isfile(path), "Cannot access: {}".format(path)

    video = io.open(path, 'r+b').read()
    encoded = base64.b64encode(video)

    display(HTML(
        data="""
        <video width="{wid}" {pt}>
        <source src="data:video/mp4;base64,{vid}" type="video/mp4" />
        </video>
        """.format(wid=width, pt=play_type, vid=encoded.decode('ascii'))
    ))


def simulate_episode(env, action_fnc, max_steps=1000, write_dir='video', remove_old_video=True, **kwargs):
    """Write and directly show video. The kwargs will be passed on to
    the show_video function.

    Args:
        env (gym.env):
        action_fnc (callable):
        max_steps (int):
        write_dir (str):
        remove_old_video (bool):
        **kwargs:
    """
    path, total_reward, n_steps = write_video(env, action_fnc, max_steps, write_dir, remove_old_video)
    print("Episode steps:\t{}".format(n_steps))
    print("Total reward:\t{:.2f}".format(total_reward))
    show_video(path, **kwargs)
