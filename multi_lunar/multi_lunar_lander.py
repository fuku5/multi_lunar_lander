import math
import numpy as np

from gym import spaces
from Box2D.b2 import edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef
from gym.envs.box2d.lunar_lander import LunarLander, ContactDetector
from gym.envs.box2d.lunar_lander \
    import FPS, SCALE, MAIN_ENGINE_POWER, SIDE_ENGINE_POWER, \
    INITIAL_RANDOM, LANDER_POLY, LEG_AWAY, LEG_DOWN, \
    LEG_W, LEG_H, LEG_SPRING_TORQUE, SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY, \
    VIEWPORT_W, VIEWPORT_H


CHUNKS = 11

class MyLunarLander(LunarLander):

    def __init__(self, start_indexes=(2, 5, 8), goal_indexes=(2, 5, 8), flatten=False, noflag=False, ground_marker=False, no_particles=False):
        self.start_indexes = start_indexes
        self.goal_indexes = goal_indexes
        self.flatten = flatten
        self.noflag = noflag
        self.ground_marker = ground_marker
        self.no_particles = no_particles
        super().__init__()
        observation_space = 8 if len(goal_indexes) == 1 else 9
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(observation_space,), dtype=np.float32)


    def reset(self, start=None, goal=None):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shapings = None
        # for debug
        self.rewards_sum = {goal: 0. for goal in self.goal_indexes}

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        self.chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]

        if self.flatten:
            height = np.ones(CHUNKS+1) * H / 4
        else:
            height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))


        self.helipad_x = []
        if goal is None:
            goal = np.random.choice(self.goal_indexes)

        self.goal = goal
        self.goal_x = self.chunk_x[goal]
        #self.goals_x = [self.chunk_x[i] for i in self.goal_indexes]
        self.goal_y = H / 4

        self.helipad_x.append(self.chunk_x[goal - 1])
        self.helipad_x.append(self.chunk_x[goal + 1])
        self.helipad_y = H / 4
        height[goal - 2] = self.helipad_y
        height[goal - 1] = self.helipad_y
        height[goal + 0] = self.helipad_y
        height[goal + 1] = self.helipad_y
        height[goal + 2] = self.helipad_y

        smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
                    for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (self.chunk_x[i],   smooth_y[i])
            p2 = (self.chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        #initial_x = VIEWPORT_W / SCALE / 2
        if start is None:
            start = np.random.choice(self.start_indexes)
        initial_x = self.chunk_x[start]
        initial_y = VIEWPORT_H / SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.lander.color1 = (0.5,0.4,0.9)
        self.lander.color2 = (0.3,0.3,0.5)
        self.lander.ApplyForceToCenter((
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        ), True)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i *
                          LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.5,0.4,0.9)
            leg.color2 = (0.3,0.3,0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                # Yes, the most esoteric numbers here, angles legs have freedom
                # to travel within
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs


        return self.step(np.array([0, 0]) if self.continuous else 0)[0]

    def step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(
                action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [
            self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                # 0.5..1.0
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            # 4 is move a bit downwards, +-2 for randomness
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + \
                side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - \
                side[1] * dispersion[1]
            impulse_pos = (self.lander.position[
                           0] + ox, self.lander.position[1] + oy)
            # particles are just a decoration, 3.5 is here to make particle
            # speed adequate
            p = self._create_particle(
                3.5, impulse_pos[0], impulse_pos[1], m_power)
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power,
                                  oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * \
                (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * \
                (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (self.lander.position[
                           0] + ox - tip[0] * 17 / SCALE, self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
            p = self._create_particle(
                0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power,
                                  oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        goal_xs = {goal: (self.chunk_x[goal] - VIEWPORT_W / SCALE / 2) / \
            (VIEWPORT_W / SCALE / 2) for goal in self.goal_indexes}
        goal_y = (self.goal_y - self.helipad_y) / (VIEWPORT_W / SCALE / 2)
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) /
            (VIEWPORT_W / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            goal_xs[self.goal], goal_y
            ][:self.observation_space.shape[0]]
        #assert len(state) == env.observation_space.shape[0]  # 8

        rewards = {goal: 0. for goal in self.goal_indexes}
        shapings = {goal: \
            - 100 * np.sqrt((state[0] - goal_xs[goal])**2 + (state[1] - goal_y)**2) \
            - 100 * np.sqrt(state[2]**2 + state[3]**2) \
            - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7] \
            for goal in self.goal_indexes}
        # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shapings is not None:
            for goal in self.goal_indexes:
                rewards[goal] = shapings[goal] - self.prev_shapings[goal]
        self.prev_shapings = shapings

        for goal in self.goal_indexes:
            rewards[goal] -= m_power * 0.30  # less fuel spent is better, about -30 for heurisic landing
            rewards[goal] -= s_power * 0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            for goal in self.goal_indexes:
                rewards[goal]= -100
        if not self.lander.awake:
            done = True
            for goal in self.goal_indexes:
                rewards[goal]= +100
        state = np.array(state)
        state.flags.writeable = False
        # for debug
        for goal in self.goal_indexes:
            self.rewards_sum[goal] += rewards[goal]
        return state, rewards[self.goal], done, {'rewards': rewards, 'rewards_sum': self.rewards_sum}

    def _draw(self):
        if __name__ == '__main__':
            import rendering
        else:
            from . import rendering
        if self.viewer is None:
            self.viewer = rendering.MyViewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE,
                                   0, VIEWPORT_H / SCALE)
        if True:
            for obj in self.particles:
                obj.ttl -= 0.15
                obj.color1 = (max(0.2, 0.2 + obj.ttl),
                              max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
                obj.color2 = (max(0.2, 0.2 + obj.ttl),
                              max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))

        self._clean_particles(self.no_particles)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(
                        f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(
                        f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(
                        path, color=obj.color2, linewidth=2)

        if not self.noflag:
            for x in self.helipad_x:
                flagy1 = self.helipad_y
                flagy2 = flagy1 + 50 / SCALE
                self.viewer.draw_polyline(
                    [(x, flagy1), (x, flagy2)], color=(1, 1, 1))
                self.viewer.draw_polygon([(x, flagy2), (x, flagy2 - 10 / SCALE),
                                          (x + 25 / SCALE, flagy2 - 5 / SCALE)], color=(0.8, 0.8, 0))
        if self.ground_marker:
            for x in self.chunk_x:
                flagy1 = self.helipad_y
                flagy2 = flagy1 + 5 / SCALE
                self.viewer.draw_polyline(
                    [(x, flagy1), (x, flagy2)], color=(1, 1, 1))






    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        self._draw()
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

def main():
    np.random.seed(100)
    game = MyLunarLander()
    game.seed(100)

    def test(start, goal):
        assert start == 2 or start == 5 or start == 8
        assert goal == 2 or goal == 5 or goal == 8
        game.reset(start, goal)
        import time
        for i in range(100):
            state, reward, done, info = game.step(np.random.randint(3))
            print(reward, info)
            game.render()
            time.sleep(1/30) 
    
    test(2,8)
    test(2,5)
    test(5,5)
    

if __name__ == '__main__':
    main()
