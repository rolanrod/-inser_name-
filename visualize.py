import util
import pg
import env
env = env.Environment()
nn = pg.PolicyNet(10 + 9 + 1 + 1 + 1 + 8, 5, 128)
seed = 861
nn = util.load_model_checkpoint(util.get_checkpoint_path(), nn)
reinforce = pg.PolicyGradient(env, nn, seed, reward_to_go=True)

print(reinforce.evaluate(100))
reinforce.run_episode(True)