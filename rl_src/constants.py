import os


REPO_NAME = 'reinforcement_learning'
REPO_PATH = os.path.realpath(__file__).split(REPO_NAME)[0] + REPO_NAME

DIR_AGENT_STORAGE = os.path.join(REPO_PATH, 'agent_storage')
