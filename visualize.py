import util
import pg
import env
env = env.Environment()
nn = pg.PolicyNet(10 + 9 + 1 + 1 + 1, 5, 128)
seed = 40
reinforce = pg.PolicyGradient(env, nn, seed, reward_to_go=True)
util.load_model_checkpoint(util.get_checkpoint_path(), nn)

reinforce.run_episode(True)