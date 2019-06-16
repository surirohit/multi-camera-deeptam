import argparse
import copy
import math
import numpy as np
import os
import pygame
from pygame.locals import *
import cv2
from timeit import default_timer as timer
import traceback
import datetime

from minos.lib import common
from minos.config.sim_args import parse_sim_args
from minos.lib.Simulator import Simulator
from minos.lib.util.ActionTraces import ActionTraces
from minos.lib.util.StateSet import StateSet
from minos.lib.util.VideoWriter import VideoWriter
import pdb

import random
random.seed(12345678)


REPLAY_MODES = ['actions', 'positions']
VIDEO_WRITER = None
TMP_SURFS = {}
MAX_DEPTH_SENSOR = 5.0
DEPTH_SCALING_FACTOR = 5000


def blit_img_to_surf(img, surf, position=(0, 0), surf_key='*'):
    global TMP_SURFS
    if len(img.shape) == 2:  # gray (y)
        img = np.dstack([img, img, img, np.ones(img.shape, dtype=np.uint8)*255])  # y -> yyy1
    else:
        img = img[:, :, [2, 1, 0, 3]]  # bgra -> rgba
    img_shape = (img.shape[0], img.shape[1])
    TMP_SURF = TMP_SURFS.get(surf_key)
    if not TMP_SURF or TMP_SURF.get_size() != img_shape:
        # print('create new surf %dx%d' % img_shape)
        TMP_SURF = pygame.Surface(img_shape, 0, 32)
        TMP_SURFS[surf_key] = TMP_SURF
    bv = TMP_SURF.get_view("0")
    bv.write(img.tostring())
    del bv
    surf.blit(TMP_SURF, position)


def display_episode_info(episode_info, display_surf, camera_outputs, show_goals=False):
    displayed = episode_info.get('displayed',0)
    if displayed < 1:
        print('episode_info', {k: episode_info[k] for k in episode_info if k != 'goalObservations'})
        if show_goals and 'goalObservations' in episode_info:
            # NOTE: There can be multiple goals with separate goal observations for each
            # We currently just handle one
            goalObservations = episode_info['goalObservations']
            if len(goalObservations) > 0:
                # Call display_response but not write to video
                display_response(goalObservations[0], display_surf, camera_outputs, print_observation=False, write_video=False)
        episode_info['displayed'] = displayed + 1


def draw_forces(forces, display_surf, area):
    r = 5
    size = round(0.45*min(area.width, area.height)-r)
    center = area.center
    pygame.draw.rect(display_surf, (0,0,0), area, 0)  # fill with black
    # assume forces are radially positioned evenly around agent
    # TODO: Actual get force sensor positions and visualize them
    dt = -2*math.pi/forces.shape[0]
    theta = math.pi/2
    for i in range(forces.shape[0]):
        x = round(center[0] + math.cos(theta)*size)
        y = round(center[1] + math.sin(theta)*size)
        width = 0 if forces[i] else 1
        pygame.draw.circle(display_surf, (255,255,0), (x,y), r, width)
        theta += dt

def draw_offset(offset, display_surf, area, color=(0,0,255)):
    dir = (offset[0], offset[2])
    mag = math.sqrt(dir[0]*dir[0] + dir[1]*dir[1])
    if mag:
        dir = (dir[0]/mag, dir[1]/mag)
    size = round(0.45*min(area.width, area.height))
    center = area.center
    target = (round(center[0]+dir[0]*size), round(center[1]+dir[1]*size))
    pygame.draw.rect(display_surf, (0,0,0), area, 0)  # fill with black
    pygame.draw.circle(display_surf, (255,255,255), center, size, 0)
    pygame.draw.line(display_surf, color, center, target, 1)
    pygame.draw.circle(display_surf, color, target, 4, 0)

def display_response(response, display_surf, camera_outputs,
                     patch_len=30, save_toc=False, print_observation=False, write_video=False):
    global VIDEO_WRITER
    observation = response.get('observation')
    sensor_data = observation.get('sensors')
    measurements = observation.get('measurements')
    # Time of contact initialized only if needed
    toc = None
    timestamp = None
    depth_img = None
    right_image = None
    rightdepth_img = None
    left_image = None
    leftdepth_img = None

    def printable(x): return type(x) is not bytearray and type(x) is not np.ndarray
    if observation is not None and print_observation:
        simple_observations = {k: v for k, v in observation.items() if k not in ['measurements', 'sensors']}
        dicts = [simple_observations, observation.get('measurements'), observation.get('sensors')]
        for d in dicts:
            for k, v in d.items():
                if type(v) is not dict:
                    info = '%s: %s' % (k,v)
                    print(info[:75] + (info[75:] and '..'))
                else:
                    print('%s: %s' % (k, str({i: v[i] for i in v if printable(v[i])})))
        if 'forces' in sensor_data:
            print('forces: %s' % str(sensor_data['forces']['data']))
        if 'info' in response:
            print('info: %s' % str(response['info']))

    if 'offset' in camera_outputs:
        draw_offset(measurements.get('offset_to_goal'), display_surf, camera_outputs['offset']['area'])

    for obs, config in camera_outputs.items():
        if obs not in sensor_data:
            continue
        if obs == 'forces':
            draw_forces(sensor_data['forces']['data'], display_surf, config['area'])
            continue
        img = sensor_data[obs]['data']
        img_viz = sensor_data[obs].get('data_viz')
        # pdb.set_trace()
        img_small=img
        if obs == 'depth': # TODO: this is the place where we need to calculate TOF
            # depth central crop
            if save_toc:
                # Clip
                img = np.clip(img, 0.1 , MAX_DEPTH_SENSOR)
                height, width = img.shape[0], img.shape[1]
                crop_d_img = img[height//2 - patch_len//2: height//2 + patch_len//2,
                                  width//2 - patch_len//2: width//2 + patch_len//2]
                toc = np.mean(crop_d_img) # Assume the object is moving at 1m/s
                toc = round(toc, 2)
                #print("Time of contact is: {:.2f}".format(toc))

            img_small=img*(255.0 / MAX_DEPTH_SENSOR)
            img_small=img_small.astype(np.uint8)
            img *= DEPTH_SCALING_FACTOR#(255.0 / MAX_DEPTH_SENSOR) # rescaling for visualization
            img = img.astype(np.uint16)
            depth_img = img
        elif obs == 'depthright': # TODO: this is the place where we need to calculate TOF
            # depth central crop
            if save_toc:
                # Clip
                img = np.clip(img, 0.1 , MAX_DEPTH_SENSOR)
                #changed
                height, width = img.shape[0], img.shape[1]
                crop_d_img = img[height//2 - patch_len//2: height//2 + patch_len//2,
                                  width//2 - patch_len//2: width//2 + patch_len//2]
                toc = np.mean(crop_d_img) # Assume the object is moving at 1m/s
                toc = round(toc, 2)
                #print("Time of contact is: {:.2f}".format(toc))
            img_small=img*(255.0 / MAX_DEPTH_SENSOR)
            img_small=img_small.astype(np.uint8)
            img *= DEPTH_SCALING_FACTOR#(255.0 / MAX_DEPTH_SENSOR) # rescaling for visualization
            img = img.astype(np.uint16)
            rightdepth_img = img
        elif obs == 'depthleft': # TODO: this is the place where we need to calculate TOF
            # depth central crop
            if save_toc:
                # Clip
                img = np.clip(img, 0.1 , MAX_DEPTH_SENSOR)
                #changed
                height, width = img.shape[0], img.shape[1]
                crop_d_img = img[height//2 - patch_len//2: height//2 + patch_len//2,
                                  width//2 - patch_len//2: width//2 + patch_len//2]
                toc = np.mean(crop_d_img) # Assume the object is moving at 1m/s
                toc = round(toc, 2)
                #print("Time of contact is: {:.2f}".format(toc))
            img_small=img*(255.0 / MAX_DEPTH_SENSOR)
            img_small=img_small.astype(np.uint8)
            img *= DEPTH_SCALING_FACTOR#(255.0 / MAX_DEPTH_SENSOR) # rescaling for visualization
            img = img.astype(np.uint16)
            leftdepth_img = img


        elif img_viz is not None:
            img = img_viz
        blit_img_to_surf(img_small, display_surf, config.get('position'))

        # TODO: consider support for writing to video of all camera modalities together
        if obs == 'color':
            color_img = img # rgb
            if write_video and VIDEO_WRITER:
                if len(img.shape) == 2:
                    VIDEO_WRITER.add_frame(np.dstack([img, img, img]))  # yyy
                else:
                    VIDEO_WRITER.add_frame(img[:, :, :-1])  # rgb
        elif obs == 'rightcamera':
            right_image = img # rgb
            if write_video and VIDEO_WRITER:
                if len(img.shape) == 2:
                    VIDEO_WRITER.add_frame(np.dstack([img, img, img]))  # yyy
                else:
                    VIDEO_WRITER.add_frame(img[:, :, :-1])  # rgb
        elif obs == 'leftcamera':
            left_image = img # rgb
            if write_video and VIDEO_WRITER:
                if len(img.shape) == 2:
                    VIDEO_WRITER.add_frame(np.dstack([img, img, img]))  # yyy
                else:
                    VIDEO_WRITER.add_frame(img[:, :, :-1])  # rgb

    if 'audio' in sensor_data:
        audio_data = sensor_data['audio']['data']
        pygame.sndarray.make_sound(audio_data).play()
        # pygame.mixer.Sound(audio_data).play()

    return toc, left_image, color_img, right_image, leftdepth_img, depth_img, rightdepth_img

def ensure_size(display_surf, rw, rh):
    w = display_surf.get_width()
    h = display_surf.get_height()
    if w < rw or h < rh:
        # Resize display (copying old stuff over)
        old_display_surf = display_surf.convert()
        display_surf = pygame.display.set_mode((max(rw,w), max(rh,h)), pygame.RESIZABLE | pygame.DOUBLEBUF)
        display_surf.blit(old_display_surf, (0,0))
        return display_surf, True
    else:
        return display_surf, False

def write_text(display_surf, text, position, font=None, fontname='monospace', fontsize=12, color=(255,255,224), align=None):
    """
    text -> string of text.
    fontname-> string having the name of the font.
    fontsize -> int, size of the font.
    color -> tuple, adhering to the color format in pygame.
    position -> tuple (x,y) coordinate of text object.args.height
    """

    font_object = font if font is not None else pygame.font.SysFont(fontname, fontsize)
    text_surface = font_object.render(text, True, color)
    if align is not None:
        text_rectangle = text_surface.get_rect()
        if align == 'center':
            text_rectangle.center = position[0], position[1]
        else:
            text_rectangle.topleft = position
        display_surf.blit(text_surface, text_rectangle)
    else:
        display_surf.blit(text_surface, position)

def interactive_loop(sim, args):
    # initialize
    pygame.mixer.pre_init(frequency=8000, channels=1)
    pygame.init()
    pygame.key.set_repeat(500, 50)  # delay, interval
    clock = pygame.time.Clock()

    now = datetime.datetime.now()

    if args.save_toc:
        tocs = []
        timestamp = []
        performed_actions = []
        agent_states = []
        last_good_action = 'idle'
        experiment_dir = os.path.join(args.save_rootdir,
                                   "{}".format(now.strftime("%Y-%m-%d_%H-%M")))
        toc_file = os.path.join(experiment_dir, "groundtruth.txt")
        image_folder = os.path.join(experiment_dir, "images")
        # create the folders if not already existent
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            os.makedirs(image_folder)

    # Set up display
    font_spacing = 20
    display_height = args.height + font_spacing*3
    all_camera_observations = ['leftcamera','color', 'rightcamera','depthleft','depth','depthright',  'normal', 'objectId', 'objectType', 'roomId', 'roomType']
    label_positions = {
        'curr': {},
        'goal': {}
    }
    camera_outputs = {
        'curr': {},
        'goal': {}
    }

    # get observation space and max height
    observation_space = sim.get_observation_space()
    spaces = [observation_space.get('sensors').get(obs) for obs in all_camera_observations if args.observations.get(obs)]
    heights = [s.shape[1] for s in spaces]

    # row with observations and goals
    nimages = 0
    total_width = 0
    max_height = max(heights)
    font_spacing = 20
    display_height = max_height + font_spacing*3
    for obs in all_camera_observations:
        if args.observations.get(obs):
            space = observation_space.get('sensors').get(obs)
            print('space', space)
            width = space.shape[0]   # TODO: have height be first to be more similar to other libraries
            height = space.shape[1]
            label_positions['curr'][obs] = (total_width, font_spacing*2)
            camera_outputs['curr'][obs] = { 'position': (total_width, font_spacing*3) }
            if args.show_goals:
                label_positions['goal'][obs] = (total_width, display_height + font_spacing*2)
                camera_outputs['goal'][obs] = { 'position': (total_width, display_height + font_spacing*3) }
            nimages += 1
            total_width += width
            if height > max_height:
                max_height = height

    if args.show_goals:
        display_height += max_height + font_spacing*3

    # Row with offset and map
    plot_size = max(min(args.height, 128), 64)
    display_height += font_spacing
    label_positions['curr']['offset'] = (0, display_height)
    camera_outputs['curr']['offset'] = { 'area': pygame.Rect(0, display_height + font_spacing, plot_size, plot_size)}

    next_start_x = plot_size
    if args.observations.get('forces'):
        label_positions['curr']['forces'] = (next_start_x, display_height)
        camera_outputs['curr']['forces'] = { 'area': pygame.Rect(next_start_x, display_height + font_spacing, plot_size, plot_size)}
        next_start_x += plot_size

    if args.observations.get('map'):
        label_positions['map'] = (next_start_x, display_height)
        camera_outputs['map'] = { 'position': (next_start_x, display_height + font_spacing) }

    display_height += font_spacing
    display_height += plot_size

    display_shape = [max(total_width, next_start_x), display_height]
    display_surf = pygame.display.set_mode((display_shape[0], display_shape[1]), pygame.RESIZABLE | pygame.DOUBLEBUF)

    # Write text
    label_positions['title'] = (display_shape[0]/2, font_spacing/2)
    write_text(display_surf, 'MINOS', fontsize=20, position = label_positions['title'], align='center')
    write_text(display_surf, 'dir_to_goal', position = label_positions['curr']['offset'])
    if args.observations.get('forces'):
        write_text(display_surf, 'forces', position = label_positions['curr']['forces'])
    if args.observations.get('map'):
        write_text(display_surf, 'map', position = label_positions['map'])
    write_text(display_surf, 'observations | controls: WASD+Arrows', position = (0, font_spacing))
    if args.show_goals:
        write_text(display_surf, 'goal', position = (0, args.height + font_spacing*3 + font_spacing))
    for obs in all_camera_observations:
        if args.observations.get(obs):
            write_text(display_surf, obs, position = label_positions['curr'][obs])
            if args.show_goals:
                write_text(display_surf, obs, position = label_positions['goal'][obs])

    # Other initialization
    scene_index = 0
    scene_dataset = args.scene.dataset

    init_time = timer()
    increment=0
    num_frames = 0
    prev_key = ''
    replay = args.replay
    action_traces = args.action_traces
    action_trace = action_traces.curr_trace() if action_traces is not None else None
    replay_auto = True
    replay_mode = args.replay_mode
    replay_mode_index = REPLAY_MODES.index(replay_mode)
    print('***\n***')
    print('CONTROLS: WASD+Arrows = move agent, R = respawn, N = next state/scene, O = print observation, Q = quit')
    if replay:
        print('P = toggle auto replay, E = toggle replay using %s '
              % str([m + ('*' if m == replay_mode else '') for m in REPLAY_MODES]))
    print('***\n***')

    prev_toc = None
    while sim.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim.running = False

        # read keys
        keys = pygame.key.get_pressed()
        print_next_observation = False
        if keys[K_q]:
            break

        if keys[K_o]:
            print_next_observation = True
        elif keys[K_n]:
            prev_key = 'n' if prev_key is not 'n' else ''
            if 'state_set' in args and prev_key is 'n':
                state = args.state_set.get_next_state()
                if not state:  # roll over to beginning
                    print('Restarting from beginning of states file...')
                    state = args.state_set.get_next_state()
                id = scene_dataset + '.' + state['scene_id']
                print('next_scene loading %s ...' % id)
                sim.set_scene(id)
                sim.move_to(state['start']['position'], state['start']['angle'])
                sim.episode_info = sim.start()
            elif prev_key is 'n':
                scene_index = (scene_index + 1) % len(args.scene_ids)
                scene_id = args.scene_ids[scene_index]
                id = scene_dataset + '.' + scene_id
                print('next_scene loading %s ...' % id)
                sim.set_scene(id)
                sim.episode_info = sim.start()
        elif keys[K_r]:
            prev_key = 'r' if prev_key is not 'r' else ''
            if prev_key is 'r':
                sim.episode_info = sim.reset()
        else:
            # Figure out action
            action = {'name': 'idle', 'strength': 50, 'angle': math.radians(5)}
            actions = []
            action_names = []
            if replay:
                unprocessed_keypressed = any(keys)
                if keys[K_p]:
                    prev_key = 'p' if prev_key is not 'p' else ''
                    if prev_key == 'p':
                        replay_auto = not replay_auto
                        unprocessed_keypressed = False
                elif keys[K_e]:
                    prev_key = 'e' if prev_key is not 'e' else ''
                    if prev_key == 'e':
                        replay_mode_index = (replay_mode_index + 1) % len(REPLAY_MODES)
                        replay_mode = REPLAY_MODES[replay_mode_index]
                        unprocessed_keypressed = False
                        print('Replay using %s' % replay_mode)

                if replay_auto or unprocessed_keypressed:
                    # get next action and do it
                    rec = action_trace.next_action_record()

                    if rec is None:
                        break
                        # go to next trace
                        #action_trace = action_traces.next_trace()
                        #start_state = action_trace.start_state()
                        #print('start_state', start_state)
                        #sim.configure(start_state)
                        #sim.episode_info = sim.start()
                    else:
                        if replay_mode == 'actions':
                            actnames = rec['actions'].split('+')
                            for actname in actnames:
                                if actname != 'reset':
                                    act = copy.copy(action)
                                    act['name'] = actname
                                    actions.append(act)
                        elif replay_mode == 'positions':
                            sim.move_to([rec['px'], rec['py'], rec['pz']], rec['rotation'])
                            actnames = rec['actions'].split('+')
                            for actname in actnames:
                                if actname != 'reset':
                                    action_names.append(actname)
                                if actname == 'lookUp' or actname == 'lookDown':
                                    act = copy.copy(action)
                                    act['name'] = actname
                                    actions.append(act)

            else:
                if keys[K_w]:
                    action['name'] = 'forwards'
                elif keys[K_s]:
                    action['name'] = 'backwards'
                elif keys[K_LEFT]:
                    action['name'] = 'turnLeft'
                elif keys[K_RIGHT]:
                    action['name'] = 'turnRight'
                elif keys[K_a]:
                    action['name'] = 'strafeLeft'
                elif keys[K_d]:
                    action['name'] = 'strafeRight'
                elif keys[K_UP]:
                    action['name'] = 'lookUp'
                elif keys[K_DOWN]:
                    action['name'] = 'lookDown'
                else:
                    action['name'] = 'idle'
                actions = [action]

        # step simulator and get observation
        action_repeat = 1

        response = sim.step(actions, action_repeat)
        if response is None:
            break

        if len(actions) > 0 and actions[0]['name'] != 'idle':
            # This is the action that will be pushed
            last_good_action = actions[0]['name']

        display_episode_info(sim.episode_info, display_surf, camera_outputs['goal'], show_goals=args.show_goals)

        # Handle map
        observation = response.get('observation')
        map = observation.get('map')
        if map is not None:
            # TODO: handle multiple maps
            if isinstance(map, list):
                map = map[0]
            config = camera_outputs['map']
            img = map['data']
            rw = map['shape'][0] + config.get('position')[0]
            rh = map['shape'][1] + config.get('position')[1]
            display_surf, resized = ensure_size(display_surf, rw, rh)
            if resized:
                write_text(display_surf, 'map', position = label_positions['map'])
            blit_img_to_surf(img, display_surf, config.get('position'), surf_key='map')

        # Handle other response
        toc, left_image, color_img, right_image, leftdepth_img, depth_img, rightdepth_img = display_response(response, display_surf, camera_outputs['curr'],
                               args.patch_len, args.save_toc,
                               print_observation=print_next_observation, write_video=True)
        if toc and prev_toc != toc and toc>0:# and last_good_action != 'idle':
            frame_fname = os.path.join(image_folder, "colorcenter_{:05d}.png".format(len(tocs)))
            right_frame_fname = os.path.join(image_folder, "colorright_{:05d}.png".format(len(tocs)))
            left_frame_fname = os.path.join(image_folder, "colorleft_{:05d}.png".format(len(tocs)))
            depth_fname = os.path.join(image_folder, "depthcenter_{:05d}.png".format(len(tocs)))
            rightdepth_fname = os.path.join(image_folder, "depthright_{:05d}.png".format(len(tocs)))
            leftdepth_fname = os.path.join(image_folder, "depthleft_{:05d}.png".format(len(tocs)))
            # Save image and append toc

            temp_width = color_img.shape[0]
            temp_height = color_img.shape[1]
            color_img = np.reshape(color_img, [temp_height, temp_width, 4])
            left_image= np.reshape(left_image, [temp_height, temp_width, 4])
            right_image= np.reshape(right_image, [temp_height, temp_width, 4])
            leftdepth_img = np.reshape(leftdepth_img, [temp_height, temp_width])
            rightdepth_img = np.reshape(rightdepth_img,[temp_height, temp_width])
            depth_img = np.reshape(depth_img,[temp_height, temp_width])




            cv2.imwrite(frame_fname,
                        cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(right_frame_fname,
                        cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(left_frame_fname,
                        cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
            # Check
            # cv2.imwrite(frame_fname,color_img)
            # cv2.imwrite(right_frame_fname,color_img)
            tocs.append(toc)
            timestamp.append(increment)
            increment=increment+1
            performed_actions.append(convert_to_number(last_good_action))
            agent_states.append(convert_to_vector(response['info']['agent_state']))
            prev_toc = toc
            # Save depth image
            #modify depth
            # depth_img *=DEPTH_SCALER
            # rightdepth_img *=DEPTH_SCALER
            # depth_img = depth_img.astype(np.uint16)
            # rightdepth_img = rightdepth_img.astype(np.uint16)

            cv2.imwrite(depth_fname, depth_img)
            cv2.imwrite(rightdepth_fname, rightdepth_img)
            cv2.imwrite(leftdepth_fname, leftdepth_img)
            #print("saving!")
        pygame.display.flip()
        num_frames += 1
        clock.tick(30)  # constraint to max 30 fps

    # NOTE: log_action_trace handled by javascript side
    # if args.log_action_trace:
    #     trace = sim.get_action_trace()
    #     print(trace['data'])

    # save tocs if necessary
    if args.save_toc:
        # concatenated_array = np.array([tocs, performed_actions]).T
        concatenated_array = np.array([timestamp]).T
        np_agent_states = np.array(agent_states)
        # pdb.set_trace()
        # for i in range (1,np_agent_states.shape[0]):
        np_agent_states[:,0:3]=np_agent_states[:,0:3]-np_agent_states[0,0:3]

        #Quaternion substraction will be added TODO
        concatenated_array = np.concatenate((concatenated_array, np_agent_states), axis=1)
        np.savetxt(toc_file, concatenated_array, fmt='%.3f',
                   header=" Timestamps tx, ty, tz, qx, qy, qz, qw")
        print("Saved {} images".format(len(tocs)))

    # cleanup and quit
    time_taken = timer() - init_time
    print('time=%f sec, fps=%f' % (time_taken, num_frames / time_taken))
    print('Thank you for playing - Goodbye!')
    pygame.quit()


def convert_to_number(action_name):
    mapper = {
        'forwards':    0,
        'backwards':   1,
        'turnLeft':    2,
        'turnRight':   3,
        'strafeLeft':  4,
        'strafeRight': 5,
        'lookUp':      6,
        'lookDown':    7,
        'idle':        8
    }

    action_number = mapper.get(action_name)
    if action_number is None:
        raise IOError("Not found action {}".format(action_name))
    return action_number

def convert_to_vector(agent_state):
    position = agent_state['position']
    orientation = agent_state['orientation']
    yaw=np.arctan2(orientation[0],orientation[2])
    rotatedposition=[-1*position[2],-1*position[0],position[1]]

    # quat=quat[0]
    print("pos ={}".format(rotatedposition))
    quat=([0,0,0,0])
    quat[3]=np.cos(yaw/2.0)
    quat[2]=np.sin(yaw/2.0)
    # pdb.set_trace()
    print("quat ={}".format(quat))
    # Returns concatenated vector
    return rotatedposition + quat

def main():
    global VIDEO_WRITER
    parser = argparse.ArgumentParser(description='Interactive interface to Simulator')
    parser.add_argument('--navmap', action='store_true',
                        default=False,
                        help='Use navigation map')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--state_set_file',
                       help='State set file')
    group.add_argument('--replay',
                       help='Load and replay action trace from file')
    parser.add_argument('--rightcamera',
                       default='True',
                       help='Right camera setting')
    parser.add_argument('--depthright',
                       default='True',
                       help='Right camera setting')
    parser.add_argument('--leftcamera',
                       default='True',
                       help='Left camera setting')
    parser.add_argument('--depthleft',
                       default='True',
                       help='Left camera setting')
    parser.add_argument('--replay_mode',
                       choices=REPLAY_MODES,
                       default='positions',
                       help='Use actions or positions for replay')
    group.add_argument('--show_goals', action='store_true',
                       default=False,
                       help='show goal observations')
    parser.add_argument('--patch_len', action='store_true',
                       default=30,
                       help='patch_length used to calculate toc in center of the image')
    parser.add_argument('--save_toc', action='store_true',
                       default=True,
                       help='associates images with time of contact')
    parser.add_argument('--save_rootdir',
                       default="/media/fadhil/ThirdPartition/3_3DVision/2_Dataset/minosdata/",
                       help='save rootdir for the generated dataset')

    args = parse_sim_args(parser)
    args.visualize_sensors = True
    sim = Simulator(vars(args))
    common.attach_exit_handler(sim)

    if 'state_set_file' in args and args.state_set_file is not None:
        args.state_set = StateSet(args.state_set_file, 1)
    if 'save_video' in args and len(args.save_video):
        filename = args.save_video if type(args.save_video) is str else 'out.mp4'
        is_rgb = args.color_encoding == 'rgba'
        VIDEO_WRITER = VideoWriter(filename, framerate=24, resolution=(args.width, args.height), rgb=is_rgb)
    if 'replay' in args and args.replay is not None:
        print('Initializing simulator using action traces %s...' % args.replay)
        args.action_traces = ActionTraces(args.replay)
        action_trace = args.action_traces.next_trace()
        sim.init()
        sim.seed(random.randint(0, 12345678))
        start_state = action_trace.start_state()
        print('start_state', start_state)
        sim.configure(start_state)
    else:
        args.action_traces = None
        args.replay = None

    try:
        print('Starting simulator...')
        ep_info = sim.start()
        if ep_info:
            print('observation_space', sim.get_observation_space())
            sim.episode_info = ep_info
            print('Simulator started.')
            interactive_loop(sim, args)
    except:
        traceback.print_exc()
        print('Error running simulator. Aborting.')

    if sim is not None:
        sim.kill()
        del sim

    if VIDEO_WRITER is not None:
        VIDEO_WRITER.close()


if __name__ == "__main__":
    main()
