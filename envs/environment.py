import os, sys
import gym
import gym.spaces as gspc
import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt

# Add PLE path
sys.path.append(os.path.join(os.path.expanduser('~'), 'src', 'PyGame-Learning-Environment'))
print(sys.path)
from ple.ple import PLE

PIPE_GAP = 90


def process_state(state):
    return np.array(list(state.values()))  # np.array([ .... ])


class PLEEnv_state(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', display_screen=True, nonstationary=False, nrandfeatures=0):
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        # open up a game state to communicate with emulator
        import importlib
        game_module_name = ('ple.games.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        self.game = getattr(game_module, game_name)(width=286, pipe_gap=PIPE_GAP)
        self.game_state = PLE(self.game, fps=30, display_screen=display_screen, state_preprocessor=process_state)
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = gspc.Discrete(len(self._action_set))
        self.screen_height, self.screen_width = self.game_state.getScreenDims()
        if game_name == 'FlappyBird':
            self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        else:
            self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(7+nrandfeatures,), dtype=np.float32)
        self.viewer = None

        # nonstationarities
        self.random_gravity = None
        self.random_speed = None
        self.gNS = False  # TODO think about how to integrate all nonstationarities
        self.hNS = False
        self.sNS = False
        self.fNS = False

        self.nonstationary = nonstationary

        # n rand features
        self.nrandfeat = nrandfeatures

        def step(a):
            if self.hNS:
                # a = decay(a)  # TODO implement function, that recursively computes the the resulting flap power. (0-1)
                pass  # TODO: pass real value to act function --> it is then treated in environment
            reward = self.game_state.act(self._action_set[a])
            state = self.game_state.getGameState()
            # TODO extend state by random uninformative feature(s), do that in all other step functions
            terminal = self.game_state.game_over()
            return state, reward, terminal, {}

        def sfNS_step(a):
            reward = self.game_state.act(self._action_set[a])
            state = self.game_state.getGameState()
            terminal = self.game_state.game_over()

            if terminal:
                # Do we need to generate new samples?
                if (len(self.random_gravity.points_list) < 1) or (len(self.random_speed.points_list) < 1):
                    self.random_gravity.add_random_points()
                    self.random_speed.add_random_points()

                # Set GRAVITY and speed of new episode here
                gravity = self.random_gravity.points_list.pop()
                self._update_gravity(gravity)
                background_speed = self.random_speed.points_list.pop()
                self._update_background_speed(background_speed)

        def sNS_step(a):
            reward = self.game_state.act(self._action_set[a])
            state = self.game_state.getGameState()
            terminal = self.game_state.game_over()

            if terminal:
                # Do we need to generate new samples?
                if (len(self.random_gravity.points_list) < 1) or (len(self.random_speed.points_list) < 1):
                    self.random_gravity.add_random_points()

                # Set GRAVITY and speed of new episode here
                gravity = self.random_gravity.points_list.pop()
                self._update_gravity(gravity)

        def fNS_step(a):
            reward = self.game_state.act(self._action_set[a])
            state = self.game_state.getGameState()
            terminal = self.game_state.game_over()

            if terminal:
                # Do we need to generate new samples?
                if (len(self.random_gravity.points_list) < 1) or (len(self.random_speed.points_list) < 1):
                    self.random_speed.add_random_points()

                # Set GRAVITY and speed of new episode here
                background_speed = self.random_speed.points_list.pop()
                self._update_background_speed(background_speed)

            return state, reward, terminal, {}

        if nonstationary == 'sNS':
            self.sNS = True
            self.fNS = False
            self.step = sNS_step

            # which parameter?
            self.update_param = self._update_gravity
            # slow or fast?
            self.param_traj = Overlayed_RandomSines(nsamples=5000, offset=1., amplitude=0.1, fband=[0.006, 0.01])
            self.param_traj.add_random_points()
        elif nonstationary == 'fNS':
            self.sNS = False
            self.fNS = True
            self.step = fNS_step
        elif nonstationary == 'sfNS':
            self.sNS = True
            self.fNS = True
            self.step = sfNS_step
        else:
            self.step = step

        # if self.nonstationary:
        #     self.step = non_stat_step
        # else:
        # self.step = step

    def _get_image(self):
        image_rotated = np.fliplr(
            np.rot90(self.game_state.getScreenRGB(), 3))  # Hack to fix the rotated image returned by ple
        return image_rotated

    @property
    def _n_actions(self):
        return len(self._action_set)

    def reset(self):
        self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.game_state.reset_game()
        state = self.game_state.getGameState()
        return state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)


    def seed(self, seed=None):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng
        self.game_state.init()
        if self.sNS:
            self.random_gravity = Overlayed_RandomSines(nsamples=5000, offset=1., amplitude=0.1, fband=[0.006, 0.01])
            self.random_gravity.add_random_points()
        if self.fNS:
            self.random_speed = RandomSteps(nsamples=5000, time_interval=[5, 20],
                                            value_interval=[3, 6])  # upper bound is excluded  # TODO set ttrace_length
            self.random_speed.add_random_points()

    def _update_gravity(self, gravity):  # Either this or flap power
        self.game.set_gravity(gravity)

    def _update_background_speed(self, speed):
        self.game.set_speed(speed)


class Filtered_RandomWalk():
    def __init__(self, nsamples, offset, amplitude, fband):
        self.nsamples = nsamples
        self.points_list = []
        self.filter_order = 4
        self.fband = np.array(fband, dtype=float)
        self.amplitude = amplitude
        self.offset = offset

    def add_random_points(self):
        # Create random data which has mean 0 and std = amplitude
        points = list(np.random.rand(self.nsamples) * self.amplitude - 0.5*self.amplitude)  # Numbers between 0-1,
        b,a = spsig.butter(self.filter_order, self.fband, btype='bandpass', analog=False, output='ba')
        points = list(spsig.lfilter(b,a, points) + self.offset)
        self.points_list = points
        # plt.figure()
        # plt.plot(self.points_list, 'r')
        # plt.show()

    def get_next_value(self):
        return self.points_list.pop()


class Overlayed_RandomSines():
    def __init__(self, nsamples, offset, amplitude, fband):
        self.nsamples = nsamples
        self.nsines = 30
        self.points_list = []
        self.fband = np.array(fband, dtype=float)
        self.amplitude = amplitude / self.nsines
        print(self.amplitude)
        self.offset = offset


    def add_random_points(self):
        # Overlay 30 different sinus curves with each having a random phase and frequency:
        sample = np.linspace(0, self.nsamples)
        phases = np.random.rand(self.nsines)
        frequencies = self.fband[0] + np.random.rand(self.nsines) * (self.fband[1] - self.fband[0])
        amplitudes = (np.random.rand(self.nsines)-0.5) * self.amplitude
        # print(amplitudes)

        # y = A * sin(2ft * pi - phase_rad)
        sin_func = lambda x, ampl, frq, ph: ampl * np.sin(2. * np.pi * (frq * x - ph))

        sine_waves = []
        plt.figure()
        for i in range(self.nsines):
            wave = [sin_func(x, amplitudes[i], frequencies[i], phases[i]) for x in np.linspace(0, self.nsamples, self.nsamples)]
            plt.plot(wave)
            sine_waves.append(wave)

        self.points_list = list(np.sum(sine_waves, axis=0))
        # plt.figure()
        # plt.plot(self.points_list)
        # plt.ylabel('gravity')
        # plt.xlabel('episode index')
        # plt.show()

    def get_next_value(self):
        return self.points_list.pop()


class RandomSteps():
    def __init__(self, nsamples, time_interval, value_interval):
        self.nsamples = nsamples
        self.points_list = []
        self.time_interval = time_interval
        self.value_interval = value_interval

    def add_random_points(self):
        # create random switching time points (switch after 5 to 20 episodes)
        repetitions = list(np.random.randint(low=self.time_interval[0], high=self.time_interval[1],
                                             size=int(self.nsamples / np.mean(self.time_interval))))
        # create values of each step
        values = list(np.random.randint(low=self.value_interval[0], high=self.value_interval[1],
                                        size=int(self.nsamples / np.mean(self.time_interval))))

        points = []
        for i in range(len(repetitions)):
            points += [values[i] for _ in range(repetitions[i])]
        self.points_list = points

        # plt.figure()
        # plt.plot(self.points_list, 'r')
        # plt.ylabel('background speed')
        # plt.xlabel('episode index')
        # plt.show()
#

# class PLEEnv_nonstat(gym.Env):
#     metadata = {'render.modes': ['human', 'rgb_array']}
#
#     def __init__(self, game_name='FlappyBird', display_screen=True):
#         # set headless mode
#         os.environ['SDL_VIDEODRIVER'] = 'dummy'
#
#         # open up a game state to communicate with emulator
#         import importlib
#         game_module_name = ('ple.games.%s' % game_name).lower()
#         game_module = importlib.import_module(game_module_name)
#         self.game = getattr(game_module, game_name)(pipe_gap=PIPE_GAP)
#         self.game_state = PLE(self.game, fps=30, display_screen=display_screen, state_preprocessor=process_state)
#         self.game_state.init()
#         # self.random_gravity = None
#         # # self.simulator = Simulator
#         self._action_set = self.game_state.getActionSet()
#         self.action_space = gspc.Discrete(len(self._action_set))
#         self.screen_height, self.screen_width = self.game_state.getScreenDims()
#         self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
#         self.viewer = None
#
#     def step(self, a):
#         reward = self.game_state.act(self._action_set[a])
#         # reward += 0.1
#
#         state = self.game_state.getGameState()
#         terminal = self.game_state.game_over()
#
#         if terminal:
#             # Do we need to generate new samples?
#             if (len(self.random_gravity.points_list) < 1) or (len(self.random_speed.points_list) < 1):
#                 self.random_gravity.add_random_points()
#                 self.random_speed.add_random_points()
#
#             # Set GRAVITY and speed here
#             gravity = self.random_gravity.points_list.pop()
#             self._update_gravity(gravity)
#
#             background_speed = self.random_speed.points_list.pop()
#             self._update_background_speed(background_speed)
#
#         return state, reward, terminal, {}
#
#     def _get_image(self):
#         image_rotated = np.fliplr(
#             np.rot90(self.game_state.getScreenRGB(), 3))  # Hack to fix the rotated image returned by ple
#         return image_rotated
#
#     @property
#     def _n_actions(self):
#         return len(self._action_set)
#
#     def reset(self):
#         self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
#         self.game_state.reset_game()
#         state = self.game_state.getGameState()
#         return state
#
#     def render(self, mode='human', close=False):
#         if close:
#             if self.viewer is not None:
#                 self.viewer.close()
#                 self.viewer = None
#             return
#         img = self._get_image()
#         if mode == 'rgb_array':
#             return img
#         elif mode == 'human':
#             from gym.envs.classic_control import rendering
#             if self.viewer is None:
#                 self.viewer = rendering.SimpleImageViewer()
#             self.viewer.imshow(img)
#
#     def seed(self, seed=None, **kwargs):
#         rng = np.random.RandomState(seed)
#         self.game_state.rng = rng
#         self.game_state.game.rng = self.game_state.rng
#         self.game_state.init()
#         self.random_gravity = Overlayed_RandomSines(nsamples=5000, offset=1., amplitude=0.1, fband=[0.006, 0.01])
#         self.random_gravity.add_random_points()
#         self.random_speed = RandomSteps(nsamples=5000, time_interval=[5,20], value_interval=[3,6])  # upper bound is excluded  # TODO set ttrace_length
#         self.random_speed.add_random_points()
#
#     def _update_gravity(self, gravity):
#         self.game.set_gravity(gravity)
#
#     def _update_background_speed(self, speed):
#         self.game.set_speed(speed)


if __name__ ==  '__main__':
    test = Overlayed_RandomSines(nsamples=300, offset=1., amplitude=0.1, fband=[0.006, 0.01])  # 1/100 - 1/166 = 0.01 - 0.006
    test.add_random_points()

    test = Overlayed_RandomSines(nsamples=3600, offset=1., amplitude=0.1,
                                 fband=[0.006, 0.01])  # 1/100 - 1/150 = 0.01 - 0.006
    test.add_random_points()

    test2 = random_speed = RandomSteps(nsamples=300, time_interval=[5, 20],
                                       value_interval=[3, 6])  # upper bound is excluded  # TODO set ttrace_length
    test2.add_random_points()

    test2 = random_speed = RandomSteps(nsamples=3600, time_interval=[5,20], value_interval=[3,6])  # upper bound is excluded  # TODO set ttrace_length
    test2.add_random_points()


