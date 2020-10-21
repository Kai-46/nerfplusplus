from read_write_model import read_model
import numpy as np
import json
import os
from pyquaternion import Quaternion
import trimesh


def parse_tracks(colmap_images, colmap_points3D):
    all_tracks = []     # list of dicts; each dict represents a track
    all_points = []     # list of all 3D points
    view_keypoints = {} # dict of lists; each list represents the triangulated key points of a view


    for point3D_id in colmap_points3D:
        point3D = colmap_points3D[point3D_id]
        image_ids = point3D.image_ids
        point2D_idxs = point3D.point2D_idxs

        cur_track = {}
        cur_track['xyz'] = (point3D.xyz[0], point3D.xyz[1], point3D.xyz[2])
        cur_track['err'] = point3D.error.item()

        cur_track_len = len(image_ids)
        assert (cur_track_len == len(point2D_idxs))
        all_points.append(list(cur_track['xyz'] + (cur_track['err'], cur_track_len) + tuple(point3D.rgb)))

        pixels = []
        for i in range(cur_track_len):
            image = colmap_images[image_ids[i]]
            img_name = image.name
            point2D_idx = point2D_idxs[i]
            point2D = image.xys[point2D_idx]
            assert (image.point3D_ids[point2D_idx] == point3D_id)
            pixels.append((img_name, point2D[0], point2D[1]))

            if img_name not in view_keypoints:
                view_keypoints[img_name] = [(point2D[0], point2D[1]) + cur_track['xyz'] + (cur_track_len, ), ]
            else:
                view_keypoints[img_name].append((point2D[0], point2D[1]) + cur_track['xyz'] + (cur_track_len, ))

        cur_track['pixels'] = sorted(pixels, key=lambda x: x[0]) # sort pixels by the img_name
        all_tracks.append(cur_track)

    return all_tracks, all_points, view_keypoints


def parse_camera_dict(colmap_cameras, colmap_images):
    camera_dict = {}
    for image_id in colmap_images:
        image = colmap_images[image_id]

        img_name = image.name
        cam = colmap_cameras[image.camera_id]

        # print(cam)
        assert(cam.model == 'PINHOLE')

        img_size = [cam.width, cam.height]
        params = list(cam.params)
        qvec = list(image.qvec)
        tvec = list(image.tvec)

        # w, h, fx, fy, cx, cy, qvec, tvec
        # camera_dict[img_name] = img_size + params + qvec + tvec
        camera_dict[img_name] = {}
        camera_dict[img_name]['img_size'] = img_size

        fx, fy, cx, cy = params
        K = np.eye(4)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        camera_dict[img_name]['K'] = list(K.flatten())

        rot = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
        W2C = np.eye(4)
        W2C[:3, :3] = rot
        W2C[:3, 3] = np.array(tvec)
        camera_dict[img_name]['W2C'] = list(W2C.flatten())

    return camera_dict


def extract_all_to_dir(sparse_dir, out_dir, ext='.bin'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    camera_dict_file = os.path.join(out_dir, 'kai_cameras.json')
    xyz_file = os.path.join(out_dir, 'kai_points.txt')
    track_file = os.path.join(out_dir, 'kai_tracks.json')
    keypoints_file = os.path.join(out_dir, 'kai_keypoints.json')
    
    colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, ext)

    camera_dict = parse_camera_dict(colmap_cameras, colmap_images)
    with open(camera_dict_file, 'w') as fp:
        json.dump(camera_dict, fp, indent=2, sort_keys=True)

    all_tracks, all_points, view_keypoints = parse_tracks(colmap_images, colmap_points3D)
    all_points = np.array(all_points)
    np.savetxt(xyz_file, all_points, header='# format: x, y, z, reproj_err, track_len, color(RGB)', fmt='%.6f')

    mesh = trimesh.Trimesh(vertices=all_points[:, :3].astype(np.float32), 
                           vertex_colors=all_points[:, -3:].astype(np.uint8))
    mesh.export(os.path.join(out_dir, 'kai_points.ply'))

    with open(track_file, 'w') as fp:
        json.dump(all_tracks, fp)

    with open(keypoints_file, 'w') as fp:
        json.dump(view_keypoints, fp)


if __name__ == '__main__':
    mvs_dir = '/home/zhangka2/sg_render/run_mvs/scan114_train_5/colmap_mvs/mvs'
    sparse_dir = os.path.join(mvs_dir, 'sparse')
    out_dir = os.path.join(mvs_dir, 'sparse_inspect')
    extract_all_to_dir(sparse_dir, out_dir)

    xyz_file = os.path.join(out_dir, 'kai_points.txt')
    reproj_errs = np.loadtxt(xyz_file)[:, 3]
    with open(os.path.join(out_dir, 'stats.txt'), 'w') as fp:
        fp.write('reprojection errors (px) in SfM:\n')
        fp.write('    percentile    value\n')
        for a in [50, 70, 90, 99]:
            fp.write('    {}            {:.3f}\n'.format(a, np.percentile(reproj_errs, a)))

    print('reprojection errors (px) in SfM:')
    print('    percentile    value')
    for a in [50, 70, 90, 99]:
        print('    {}            {:.3f}'.format(a, np.percentile(reproj_errs, a)))