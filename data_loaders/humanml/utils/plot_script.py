import math
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap
import torch

matplotlib.use('Agg')


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, variance = None, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color 
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def draw_sphere(position, radius, color='c', alpha=0.1):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = radius * np.cos(u) * np.sin(v) + position[0]
        y = radius * np.sin(u) * np.sin(v) + position[1]
        z = radius * np.cos(v) + position[2]
        ax.plot_surface(x, y, z, color=color, alpha=alpha)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        # plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
        #              MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        specific_joints_indices = [3, 7, 8, 12, 20, 21]

        # Draw spheres around specific joints
        if variance is not None:
            for joint_idx in specific_joints_indices:
                joint_position = data[index, joint_idx]
                # joint_variance = np.exp(np.mean(variance[index, joint_idx]))  # Convert log variance to actual variance
                joint_variance = np.exp(np.mean(variance[index, joint_idx])) * 0.3
                radius = np.sqrt(joint_variance) / 3  # Simplified radius calculation
                draw_sphere(joint_position, radius, color='c', alpha=0.1)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()


# def plot_3d_motion_with_gt(save_path, kinematic_tree, joints, title, dataset, variance = None, gt_data=None, figsize=(3, 3), fps=120, radius=3,
#                    vis_mode='default', gt_frames=[], emb_motion_len=0):
#     matplotlib.use('Agg')

#     title = '\n'.join(wrap(title, 20))

#     def init():
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_ylim3d([0, radius])
#         ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
#         # print(title)
#         fig.suptitle(title, fontsize=10)
#         ax.grid(b=False)

#     def plot_xzPlane_gt(minx, maxx, miny, minz, maxz):
#         ## Plot a plane XZ
#         verts = [
#             [minx, miny, minz],
#             [minx, miny, maxz],
#             [maxx, miny, maxz],
#             [maxx, miny, minz]
#         ]
#         xz_plane = Poly3DCollection([verts])
#         xz_plane.set_facecolor((0.7, 0.5, 0.5, 0.5))
#         ax.add_collection3d(xz_plane)

#     def plot_xzPlane_pred(minx, maxx, miny, minz, maxz):
#         ## Plot a plane XZ
#         verts = [
#             [minx, miny, minz],
#             [minx, miny, maxz],
#             [maxx, miny, maxz],
#             [maxx, miny, minz]
#         ]
#         xz_plane = Poly3DCollection([verts])
#         xz_plane.set_facecolor((0.3, 0.5, 0.5, 0.5))
#         ax.add_collection3d(xz_plane)

#     data = joints.copy().reshape(len(joints), -1, 3)
#     gt_data = gt_data.copy().reshape(len(gt_data), -1, 3)

#     # preparation related to specific datasets
#     if dataset == 'kit':
#         data *= 0.003  # scale for visualization
#     elif dataset == 'humanml':
#         data *= 1.3  # scale for visualization
#         gt_data *= 1.3
#     elif dataset in ['humanact12', 'uestc']:
#         data *= -1.5 # reverse axes, scale for visualization

#     fig = plt.figure(figsize=figsize)
#     plt.tight_layout()
#     ax = p3.Axes3D(fig)
#     init()
#     MINS_pred = data.min(axis=0).min(axis=0)
#     MAXS_pred = data.max(axis=0).max(axis=0)
#     MINS_gt = gt_data.min(axis=0).min(axis=0)
#     MAXS_gt = gt_data.max(axis=0).max(axis=0)

#     colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
#     colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
#     colors = colors_orange
#     if vis_mode == 'upper_body':  # lower body taken fixed to input motion
#         colors[0] = colors_blue[0]
#         colors[1] = colors_blue[1]
#     elif vis_mode == 'gt':
#         colors = colors_blue

#     frame_number = data.shape[0]

#     height_offset_pred = MINS_pred[1]
#     height_offset_gt = MINS_gt[1]
#     data[:, :, 1] -= height_offset_pred
#     gt_data[:, :, 1] -= height_offset_gt
#     trajec_pred = data[:, 0, [0, 2]]
#     trajec_gt = gt_data[:, 0, [0, 2]]

#     data[..., 0] -= data[:, 0:1, 0]
#     data[..., 2] -= data[:, 0:1, 2]
#     gt_data[..., 0] -= gt_data[:, 0:1, 0]
#     gt_data[..., 2] -= gt_data[:, 0:1, 2]


#     def draw_sphere(position, radius, color='c', alpha=0.1):
#         u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#         x = radius * np.cos(u) * np.sin(v) + position[0]
#         y = radius * np.sin(u) * np.sin(v) + position[1]
#         z = radius * np.cos(v) + position[2]
#         ax.plot_surface(x, y, z, color=color, alpha=alpha)

#     def update(index):
#         ax.view_init(elev=120, azim=-90)
#         ax.dist = 7.5

#         plot_xzPlane_gt(MINS_gt[0] - trajec_gt[index, 0], MAXS_gt[0] - trajec_gt[index, 0], 0, MINS_gt[2] - trajec_gt[index, 1],
#                      MAXS_gt[2] - trajec_gt[index, 1])

#         for i, (chain, color_orange, color_blue) in enumerate(zip(kinematic_tree, colors_orange, colors_blue)):
#             if i < 5:
#                 linewidth = 4.0
#             else:
#                 linewidth = 2.0       
#             ax.plot3D(gt_data[index, chain, 0], gt_data[index, chain, 1], gt_data[index, chain, 2], linewidth=linewidth, color=color_blue)
        
#         plot_xzPlane_pred(MINS_pred[0] - trajec_pred[index, 0], MAXS_pred[0] - trajec_pred[index, 0], 0, MINS_pred[2] - trajec_pred[index, 1],
#                      MAXS_pred[2] - trajec_pred[index, 1])

#         for i, (chain, color_orange, color_blue) in enumerate(zip(kinematic_tree, colors_orange, colors_blue)):
#             if i < 5:
#                 linewidth = 4.0
#             else:
#                 linewidth = 2.0
#             # if index > emb_motion_len:
#             ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color_orange)
        

#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])

#         specific_joints_indices = [3, 7, 8, 12, 20, 21]

#         # Draw spheres around specific joints
#         # if index > emb_motion_len:
#         if variance is not None:
#             for joint_idx in specific_joints_indices:
#                 joint_position = data[index, joint_idx]
#                 # joint_variance = np.exp(np.mean(variance[index, joint_idx]))  # Convert log variance to actual variance
#                 joint_variance = np.exp(np.mean(variance[index, joint_idx])) * 0.3
#                 radius = np.sqrt(joint_variance) / 3  # Simplified radius calculation
#                 draw_sphere(joint_position, radius, color='c', alpha=0.1)

#     ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

#     ani.save(save_path, fps=fps)

#     plt.close()

def plot_3d_motion_with_gt(save_path, kinematic_tree, joints, title, dataset, variance=None, gt_data=None, figsize=(3, 3), fps=120, radius=3,
                           vis_mode='default', gt_frames=[], emb_motion_len=0):

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
        return xz_plane

    data = joints.copy().reshape(len(joints), -1, 3)
    gt_data = gt_data.copy().reshape(len(gt_data), -1, 3)

    if dataset == 'kit':
        data *= 0.003
    elif dataset == 'humanml':
        data *= 1.3
        gt_data *= 1.3
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = Axes3D(fig)
    init()
    MINS = np.minimum(data.min(axis=0).min(axis=0), gt_data.min(axis=0).min(axis=0))
    MAXS = np.maximum(data.max(axis=0).max(axis=0), gt_data.max(axis=0).max(axis=0))
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    # specific_joints_indices = [3, 7, 8, 12, 20, 21]
    specific_joints_indices = [2, 6, 7, 11, 19, 20]
    colors = colors_orange
    if vis_mode == 'upper_body':
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    gt_data[:, :, 1] -= height_offset
    trajec_pred = data[:, 0, [0, 2]]
    trajec_gt = gt_data[:, 0, [0, 2]]

    lines_gt = [ax.plot([], [], [], linewidth=4.0 if i < 5 else 2.0, color=colors_blue[i % len(colors_blue)])[0] for i in range(len(kinematic_tree))]
    lines_pred = [ax.plot([], [], [], linewidth=4.0 if i < 5 else 2.0, color=colors_orange[i % len(colors_orange)])[0] for i in range(len(kinematic_tree))]
    xz_plane_gt = plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
    xz_plane_pred = plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

    spheres = []

    def draw_sphere(position, radius, color='c', alpha=0.1):
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = radius * np.cos(u) * np.sin(v) + position[0]
        y = radius * np.sin(u) * np.sin(v) + position[1]
        z = radius * np.cos(v) + position[2]
        return ax.plot_surface(x, y, z, color=color, alpha=alpha)

    def update(index):
        ax.view_init(elev=120, azim=-90)
        ax.dist = 8.5

        # Update ground truth trajectory plane
        xz_plane_gt.set_verts([[
            [MINS[0] - trajec_gt[index, 0], 0, MINS[2] - trajec_gt[index, 1]],
            [MINS[0] - trajec_gt[index, 0], 0, MAXS[2] - trajec_gt[index, 1]],
            [MAXS[0] - trajec_gt[index, 0], 0, MAXS[2] - trajec_gt[index, 1]],
            [MAXS[0] - trajec_gt[index, 0], 0, MINS[2] - trajec_gt[index, 1]]
        ]])

        # Update predicted trajectory plane
        xz_plane_pred.set_verts([[
            [MINS[0] - trajec_pred[index, 0], 0, MINS[2] - trajec_pred[index, 1]],
            [MINS[0] - trajec_pred[index, 0], 0, MAXS[2] - trajec_pred[index, 1]],
            [MAXS[0] - trajec_pred[index, 0], 0, MAXS[2] - trajec_pred[index, 1]],
            [MAXS[0] - trajec_pred[index, 0], 0, MINS[2] - trajec_pred[index, 1]]
        ]])

        # UNCOMMENT TO ADD JOINT LABELS
        # for txt in ax.texts:
        #     txt.set_visible(False)
        # ax.texts.clear()

        for i, (chain, line_gt, line_pred) in enumerate(zip(kinematic_tree, lines_gt, lines_pred)):
            line_gt.set_data(gt_data[index, chain, 0], gt_data[index, chain, 1])
            line_gt.set_3d_properties(gt_data[index, chain, 2])
            line_pred.set_data(data[index, chain, 0], data[index, chain, 1])
            line_pred.set_3d_properties(data[index, chain, 2])

            # UNCOMMENT TO ADD JOINT LABELS
            # for joint_idx in chain:
            #     x, y, z = gt_data[index, joint_idx]
            #     ax.text(x, y, z, f'{joint_idx}', color='black', fontsize=8, ha='center', va='center')

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Remove existing spheres
        while spheres:
            sphere = spheres.pop()
            sphere.remove()

        if variance is not None:
            for joint_idx in specific_joints_indices:
                joint_position = data[index, joint_idx+1]
                joint_variance = np.exp(0.5*np.mean(variance[index, joint_idx]))
                # joint_variance_exp = np.exp(joint_variance) * 0.47
                joint_variance_exp = np.exp(joint_variance) * 0.3
                joint_variance_transformed = joint_variance_exp ** 8  # You can experiment with different powers or transformations
                radius = joint_variance_transformed / 3
                # radius = joint_variance / 3
                sphere = draw_sphere(joint_position, radius, color='c', alpha=0.1)
                spheres.append(sphere)

        # if variance is not None:
        #     for joint_idx in specific_joints_indices:
        #         joint_position = data[index, joint_idx+1]
        #         joint_variance = np.exp(0.5*np.mean(variance[index, joint_idx]))
        #         radius = joint_variance / 3
        #         sphere = draw_sphere(joint_position, radius, color='c', alpha=0.1)
        #         spheres.append(sphere)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    ani.save(save_path, fps=fps)

    plt.close()

# def plot_3d_motion_with_gt(save_path, kinematic_tree, joints, title, dataset, variance=None, gt_data=None, figsize=(3, 3), fps=120, radius=3,
#                            vis_mode='default', gt_frames=[], emb_motion_len=0):
#     import matplotlib
#     from matplotlib import pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     from matplotlib.animation import FuncAnimation
#     from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#     import numpy as np
#     from textwrap import wrap
#     from matplotlib.colors import to_rgba

#     matplotlib.use('Agg')

#     title = '\n'.join(wrap(title, 20))

#     def init():
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_ylim3d([0, radius])
#         ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
#         fig.suptitle(title, fontsize=10)
#         ax.grid(b=False)

#     def plot_xzPlane(minx, maxx, miny, minz, maxz):
#         verts = [
#             [minx, miny, minz],
#             [minx, miny, maxz],
#             [maxx, miny, maxz],
#             [maxx, miny, minz]
#         ]
#         xz_plane = Poly3DCollection([verts])
#         xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#         ax.add_collection3d(xz_plane)
#         return xz_plane

#     data = joints.copy().reshape(len(joints), -1, 3)
#     gt_data = gt_data.copy().reshape(len(gt_data), -1, 3)

#     if dataset == 'kit':
#         data *= 0.003
#     elif dataset == 'humanml':
#         data *= 1.3
#         gt_data *= 1.3
#     elif dataset in ['humanact12', 'uestc']:
#         data *= -1.5

#     fig = plt.figure(figsize=figsize)
#     plt.tight_layout()
#     ax = Axes3D(fig)
#     init()
#     MINS = gt_data.min(axis=0).min(axis=0)
#     MAXS = gt_data.max(axis=0).max(axis=0)
#     colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
#     colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
#     specific_joints_indices = [2, 6, 7, 11, 19, 20]
#     colors = colors_orange
#     if vis_mode == 'upper_body':
#         colors[0] = colors_blue[0]
#         colors[1] = colors_blue[1]
#     elif vis_mode == 'gt':
#         colors = colors_blue

#     frame_number = data.shape[0]

#     height_offset = MINS[1]
#     data[:, :, 1] -= height_offset
#     gt_data[:, :, 1] -= height_offset
#     trajec_pred = data[:, 0, [0, 2]]
#     trajec_gt = gt_data[:, 0, [0, 2]]

#     lines_gt = [ax.plot([], [], [], linewidth=4.0 if i < 5 else 2.0, color=to_rgba(colors_blue[i % len(colors_blue)], 0.1))[0] for i in range(len(kinematic_tree))]
#     lines_pred = [ax.plot([], [], [], linewidth=4.0 if i < 5 else 2.0, color=to_rgba(colors_orange[i % len(colors_orange)], 0.1))[0] for i in range(len(kinematic_tree))]
#     xz_plane_gt = plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

#     accumulated_lines_gt = []
#     accumulated_lines_pred = []

#     def update(index):
#         if index in frames_to_accumulate:
#             ax.view_init(elev=120, azim=-90)
#             ax.dist = 8.5

#             # Interpolate alpha based on index to create a fade effect
#             alpha = 0.1 + 0.9 * (index / max(frames_to_accumulate))

#             for i, (chain, line_gt, line_pred) in enumerate(zip(kinematic_tree, lines_gt, lines_pred)):
#                 line_gt, = ax.plot(gt_data[index, chain, 0], gt_data[index, chain, 1], gt_data[index, chain, 2], linewidth=4.0 if i < 5 else 2.0, color=to_rgba(colors_blue[i % len(colors_blue)], alpha))
#                 line_pred, = ax.plot(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=4.0 if i < 5 else 2.0, color=to_rgba(colors_orange[i % len(colors_orange)], alpha))
#                 accumulated_lines_gt.append(line_gt)
#                 accumulated_lines_pred.append(line_pred)

#             plt.axis('off')
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#             ax.set_zticklabels([])

#     frames_to_accumulate = range(0, min(115, frame_number), 15)
#     ani = FuncAnimation(fig, update, frames=frames_to_accumulate, interval=1000 / fps, repeat=False)

#     ani.save(save_path, fps=fps)

#     plt.close()

