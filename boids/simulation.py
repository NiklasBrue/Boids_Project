# -*- coding: utf-8 -*-

import numpy as np
import random
from random import choice
import pyglet
from pyglet.gl import (
    Config, glScalef,
    glEnable, glBlendFunc, glLoadIdentity, glClearColor,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_COLOR_BUFFER_BIT)
from pyglet.window import key

from .boid import Boid
from .predator import Predator
from .attractor import Attractor
from .obstacle import Obstacle
from .vector import distance

def create_random_obstacle(width, height, mask, max_size):
    position = np.array([random.uniform(0, width), random.uniform(0, height)])
    # position = np.array([100, random.uniform(0,height)])
    size=random.uniform(0, max_size)
    obj = Obstacle(position=position, size=size)
    mask = cut_from_mask(mask, position, size)
    return obj, mask

def cut_from_mask(mask, position, size):
    """ Mask is set over the hole are to avoid boids initialized in object"""
    width = mask.shape[0]
    height = mask.shape[1]
    position = np.asarray(position).astype(int)
    size = np.round(size+1).astype(int)
    grid = np.moveaxis(np.mgrid[:width,:height], 0, -1)
    dist = np.linalg.norm(np.concatenate(grid - position),axis=1)
    dist = dist.reshape([width, height])
    mask[dist<size] = 0
    return mask

def create_random_boid(mask, scale_domain):
    width = mask.shape[0]
    height = mask.shape[1]
    #print("# DEBUG: bounds: ", [width, height])
    mask = mask.astype(bool)
    grid = np.moveaxis(np.mgrid[:width,:height], 0, -1)
    gridpoints = grid[mask]
    select_grid_point = int(random.random()*gridpoints.shape[0])
    position = gridpoints[select_grid_point] + np.array([random.random(),\
                            random.random()])
    return Boid(
        position=[random.uniform(0, width), random.uniform(0, height)],
        bounds=[width, height],
        velocity=np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0,\
                            50.0)]),
        # velocity = [0,0],
        scale_domain=scale_domain,
        age=random.uniform(0,1))
        # color=[random.random(), random.random(), random.random()])

def create_centered_boid_flock(mask, scale_domain):
    width = mask.shape[0]
    height = mask.shape[1]
    #print("# DEBUG: bounds: ", [width, height])
    mask = mask.astype(bool)
    grid = np.moveaxis(np.mgrid[:width,:height], 0, -1)
    gridpoints = grid[mask]
    select_grid_point = int(random.random()*gridpoints.shape[0])
    position = gridpoints[select_grid_point] + np.array([random.random(),\
                            random.random()])
    return Boid(
        position=[random.uniform(3*width/8, 5*width/8), random.uniform(3*height/8, 5*height/8)],
        bounds=[width, height],
        velocity=np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0,\
                            50.0)]),
        scale_domain=scale_domain,
        age=random.uniform(0,1))

def create_random_predator(mask, scale_domain):
    width = mask.shape[0]
    height = mask.shape[1]
    #print("# DEBUG: bounds: ", [width, height])
    mask = mask.astype(bool)
    grid = np.moveaxis(np.mgrid[:width,:height], 0, -1)
    gridpoints = grid[mask]
    select_grid_point = int(random.random()*gridpoints.shape[0])
    position = gridpoints[select_grid_point] + np.array([random.random(),\
                            random.random()])
    return Predator(
        position=[random.uniform(0, width), random.uniform(0, height)],
        bounds=[width, height],
        velocity=np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0,\
                            50.0)]),
        # velocity = [0,0],
        scale_domain=scale_domain)

def create_predator_circle(mask, scale_domain):
    width = mask.shape[0]
    height = mask.shape[1]
    mask = mask.astype(bool)
    grid = np.moveaxis(np.mgrid[:width,:height], 0, -1)
    gridpoints = grid[mask]
    select_grid_point = int(random.random()*gridpoints.shape[0])
    position = gridpoints[select_grid_point] + np.array([random.random(),\
                            random.random()])
    choice_x = choice([(width/8, 2*width/8), (6*width/8, 7*width/8)])
    choice_y = choice([(height/8, 2*height/8), (6*height/8, 7*height/8)])
    return Predator(
        position=[random.uniform(*choice_x), random.uniform(*choice_y)],
        bounds=[width, height],
        velocity=np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0,\
                            50.0)]),
        scale_domain=scale_domain)

def delete_catched_boid(boids, predators):
    removed_boids = []
    for boid in boids:
        for predator in predators:
            dist = distance(boid.position, predator.position)
            if dist <= 2*boid.size:
                removed_boids.append(boid)
    return removed_boids

def get_window_config():
    platform = pyglet.window.get_platform()
    display = platform.get_default_display()
    screen = display.get_default_screen()

    template = Config(double_buffer=True, sample_buffers=1, samples=4)
    try:
        config = screen.get_best_config(template)
    except pyglet.window.NoSuchConfigException:
        template = Config()
        config = screen.get_best_config(template)

    return config


def run():
    scale_domain = 2 # area is virtually scaled by decreasing other stuff

    glScalef(0.5, 0.5, 0.5)
    show_debug = False
    show_vectors = False

    # window_width = 4000
    # window_height = 4000

    n_boids = 20
    boids = []

    n_predators = 4
    predators = []

    attractors = []

    n_obstacles = 50
    max_size = 70/scale_domain
    obstacles = []

    mouse_location = (0, 0)
    window = pyglet.window.Window(
        # width = 4000,
        # height = 4000,
        # resizable  = True,
        fullscreen=True,
        caption="Boids Simulation",
        config=get_window_config())

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    #window.push_handlers(pyglet.window.event.WindowEventLogger())
    # window.set_size(window_width, window_height)
    # window.set_pixel_ratio(2)
    #print("# DEBUG: window.width: ", window.width)
    #print("# DEBUG: window.height: ", window.height)
    mask = np.ones([window.width, window.height])

    for i in range(1, n_obstacles):
        obj, mask = create_random_obstacle(window.width, window.height, mask,\
                                            max_size)
        obstacles.append(obj)

    for i in range(n_boids):
        boids.append(create_centered_boid_flock(mask, scale_domain))

    for i in range(n_predators):
        predators.append(create_predator_circle(mask, scale_domain))

    deleted_boids = []

    def update(dt):
        boids_to_delete = delete_catched_boid(boids, predators)
        deleted_boids.append(boids_to_delete)
        for boid in boids_to_delete:
            if boid in boids: #ensures that if a boid is catched by 2 predators no errors are produced
                boids.remove(boid)
                print(boid)
        for boid in boids:
            boid.update(dt, boids, attractors, obstacles, predators)
        for predator in predators:
            predator.update(dt, predators, boids, obstacles)
        

    # schedule world updates as often as possible
    pyglet.clock.schedule(update)

    @window.event
    def on_draw():
        glClearColor(0.1, 0.1, 0.1, 1.0)
        window.clear()
        glLoadIdentity()

        for boid in boids:
            boid.draw(show_velocity=show_debug, show_view=show_debug, show_vectors=show_vectors)

        for predator in predators:
            predator.draw(show_velocity=show_debug, show_view=show_debug, show_vectors=show_vectors)

        for attractor in attractors:
            attractor.draw()

        for obstacle in obstacles:
            obstacle.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.Q:
            pyglet.app.exit()
        # elif symbol == key.EQUAL and modifiers & key.MOD_SHIFT:
        #     boids.append(create_random_boid(window.width, window.height))
        elif symbol == key.MINUS and len(boids) > 0:
            boids.pop()
        elif symbol == key.D:
            nonlocal show_debug
            show_debug = not show_debug
        elif symbol == key.V:
            nonlocal show_vectors
            show_vectors = not show_vectors
        elif symbol == key.A:
            attractors.append(Attractor(position=mouse_location))
        elif symbol == key.O:
            obstacles.append(Obstacle(position=mouse_location))

    @window.event
    def on_mouse_drag(x, y, *args):
        nonlocal mouse_location
        mouse_location = x, y

    @window.event
    def on_mouse_motion(x, y, *args):
        nonlocal mouse_location
        mouse_location = x, y

    pyglet.app.run()
