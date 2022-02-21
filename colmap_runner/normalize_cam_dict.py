import numpy as np
import json
import copy
import open3d as o3d


def get_tf_cams(cam_dict, target_radius=1.):
    cam_centers = []
    for im_name in cam_dict:
        W2C = np.array(cam_dict[im_name]['W2C']).reshape((4, 4))
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    scale = target_radius / radius

    return translate, scale


def normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, target_radius=1., in_geometry_file=None, out_geometry_file=None):
    with open(in_cam_dict_file) as fp:
        in_cam_dict = json.load(fp)

    translate, scale = get_tf_cams(in_cam_dict, target_radius=target_radius)

    if in_geometry_file is not None and out_geometry_file is not None:
        # check this page if you encounter issue in file io: http://www.open3d.org/docs/0.9.0/tutorial/Basic/file_io.html
        geometry = o3d.io.read_triangle_mesh(in_geometry_file)
        
        tf_translate = np.eye(4)
        tf_translate[:3, 3:4] = translate
        tf_scale = np.eye(4)
        tf_scale[:3, :3] *= scale
        tf = np.matmul(tf_scale, tf_translate)

        geometry_norm = geometry.transform(tf)
        o3d.io.write_triangle_mesh(out_geometry_file, geometry_norm)
  
    def transform_pose(W2C, translate, scale):
        C2W = np.linalg.inv(W2C)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        return np.linalg.inv(C2W)

    out_cam_dict = copy.deepcopy(in_cam_dict)
    for img_name in out_cam_dict:
        W2C = np.array(out_cam_dict[img_name]['W2C']).reshape((4, 4))
        W2C = transform_pose(W2C, translate, scale)
        assert(np.isclose(np.linalg.det(W2C[:3, :3]), 1.))
        out_cam_dict[img_name]['W2C'] = list(W2C.flatten())

    with open(out_cam_dict_file, 'w') as fp:
        json.dump(out_cam_dict, fp, indent=2, sort_keys=True)


if __name__ == '__main__':
    in_cam_dict_file = ''
    out_cam_dict_file = ''
    normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, target_radius=1.)
