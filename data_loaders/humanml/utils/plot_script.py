import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap

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


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
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

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
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

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()

# def plot_3d_motion_with_spheres(save_path, kinematic_tree, joints, variances, title, dataset, figsize=(3, 3), fps=120, radius=3,
#                                 vis_mode='default', gt_frames=[]):
#     matplotlib.use('Agg')

#     title = '\n'.join(wrap(title, 20))

#     def init():
#         ax.set_xlim3d([-radius, radius])
#         ax.set_ylim3d([-radius, radius])
#         ax.set_zlim3d([-radius, radius])
#         fig.suptitle(title, fontsize=10)
#         ax.grid(b=False)

#     # def plot_xzPlane(minx, maxx, miny, minz, maxz):
#     #     ## Plot a plane XZ
#     #     verts = [
#     #         [minx, miny, minz],
#     #         [minx, miny, maxz],
#     #         [maxx, miny, maxz],
#     #         [maxx, miny, minz]
#     #     ]
#     #     xz_plane = Poly3DCollection([verts])
#     #     xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#     #     ax.add_collection3d(xz_plane)

#     #         return ax

#     # (seq_len, joints_num, 3)
#     data = joints.copy().reshape(len(joints), -1, 3)

#     # preparation related to specific datasets
#     if dataset == 'kit':
#         data *= 0.003  # scale for visualization
#     elif dataset == 'humanml':
#         data *= 1.3  # scale for visualization
#     elif dataset in ['humanact12', 'uestc']:
#         data *= -1.5 # reverse axes, scale for visualization

#     fig = plt.figure(figsize=figsize)
#     plt.tight_layout()
#     ax = p3.Axes3D(fig)
#     init()
#     MINS = data.min(axis=0).min(axis=0)
#     MAXS = data.max(axis=0).max(axis=0)
#     colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
#     colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
#     colors = colors_orange
#     if vis_mode == 'upper_body':  # lower body taken fixed to input motion
#         colors[0] = colors_blue[0]
#         colors[1] = colors_blue[1]
#     elif vis_mode == 'gt':
#         colors = colors_blue

#     frame_number = data.shape[0]
#     #     print(dataset.shape)

#     height_offset = MINS[1]
#     data[:, :, 1] -= height_offset
#     trajec = data[:, 0, [0, 2]]

#     data[..., 0] -= data[:, 0:1, 0]
#     data[..., 2] -= data[:, 0:1, 2]

#     #     print(trajec.shape)

#     def update(index):
#         ax.clear()
#         init()  # Re-initialize to clear the plot
        
#         for joint, variance in zip(data[index], variances[index]):
#             _draw_ellipsoid(ax, joint, np.sqrt(variance))  # Using the sqrt of variance as the radius

#         # Optional: plot the skeleton lines
#         for chain, color in zip(kinematic_tree, colors):
#             ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], color=color)

#         # for joint in data[index]:
#         #     _draw_sphere(ax, joint, sphere_radius)

#         # # Optional: plot the skeleton lines (you can remove this part if not needed)
#         # for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
#         #     linewidth = 4.0 if i < 5 else 2.0
#         #     ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

#     ani = FuncAnimation(fig, update, frames=len(data), interval=1000 / fps, repeat=False)

#     # writer = FFMpegFileWriter(fps=fps)
#     ani.save(save_path, fps=fps)
#     # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
#     # ani.save(save_path, writer='pillow', fps=1000 / fps)

#     plt.close()

def plot_3d_motion_with_spheres(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3, sphere_radius=0.05, sphere_indices=[1, 5, 10], vis_mode='default', gt_frames=[]):
    title = '\n'.join(wrap(title, 60))  # Format title for better display
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)

    # Set the limits of the plot
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(False)

    data = joints.copy().reshape(len(joints), -1, 3)  # Reshape data for plotting

    # Function to plot a plane (used to illustrate the ground)
    def plot_xz_plane(ax, minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        ax.add_collection3d(Poly3DCollection([verts], facecolors='gray', linewidths=1, edgecolors='r', alpha=.25))

    def update(index):
        ax.clear()
        init()
        plot_xz_plane(ax, -radius / 2, radius / 2, 0, -radius / 3, radius * 2 / 3)

        # Plotting the skeleton
        for i, (chain, color) in enumerate(zip(kinematic_tree, plt.cm.tab10.colors)):
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], color=color)

        # Plotting spheres at specific joints
        for joint_index in sphere_indices:
            joint_coords = data[index, joint_index]
            ax.scatter(joint_coords[0], joint_coords[1], joint_coords[2], s=1000 * sphere_radius**2, color='red', alpha=0.6)

        plt.axis('off')

    writer = FFMpegFileWriter(fps=fps)
    ani = FuncAnimation(fig, update, frames=len(joints), interval=1000 / fps, init_func=init, repeat=False)
    ani.save(save_path, writer=writer)
    plt.close()

def _draw_ellipsoid(ax, center, radii):
    """
    Draw an ellipsoid centered at 'center' with radii specified by 'radii'.

    Args:
    - ax: The matplotlib axis to draw on.
    - center: The center of the ellipsoid (x, y, z coordinates).
    - radii: Radii along each axis (corresponds to the square root of the variance).
    """
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = radii[0] * np.cos(u) * np.sin(v)
    y = radii[1] * np.sin(u) * np.sin(v)
    z = radii[2] * np.cos(v)

    # Translate the ellipsoid to the center
    x += center[0]
    y += center[1]
    z += center[2]

    ax.plot_surface(x, y, z, color='b', alpha=0.3)