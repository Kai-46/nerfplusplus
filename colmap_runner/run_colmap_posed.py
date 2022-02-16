import os
import json
from database import COLMAPDatabase
from pyquaternion import Quaternion
import numpy as np
import imageio
import subprocess


def bash_run(cmd):
    # local install of colmap
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/home/zhangka2/code/colmap/build/__install__/lib'
    
    colmap_bin = '/home/zhangka2/code/colmap/build/__install__/bin/colmap'
    cmd = colmap_bin + ' ' + cmd
    print('\nRunning cmd: ', cmd)

    subprocess.check_call(['/bin/bash', '-c', cmd], env=env)


gpu_index = '-1'


def run_sift_matching(img_dir, db_file):
    print('Running sift matching...')

    # if os.path.exists(db_file): # otherwise colmap will skip sift matching
    #     os.remove(db_file)

    # feature extraction
    # if there's no attached display, cannot use feature extractor with GPU
    cmd = ' feature_extractor --database_path {} \
                                    --image_path {} \
                                    --ImageReader.camera_model PINHOLE \
                                    --SiftExtraction.max_image_size 5000  \
                                    --SiftExtraction.estimate_affine_shape 0 \
                                    --SiftExtraction.domain_size_pooling 1 \
                                    --SiftExtraction.num_threads 32 \
                                    --SiftExtraction.use_gpu 1 \
                                    --SiftExtraction.gpu_index {}'.format(db_file, img_dir, gpu_index)
    bash_run(cmd)

    # feature matching
    cmd = ' exhaustive_matcher --database_path {} \
                                     --SiftMatching.guided_matching 1 \
                                     --SiftMatching.use_gpu 1 \
                                     --SiftMatching.gpu_index {}'.format(db_file, gpu_index)

    bash_run(cmd)


def create_init_files(pinhole_dict_file, db_file, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create template
    with open(pinhole_dict_file) as fp:
        pinhole_dict = json.load(fp)

    template = {}
    cameras_line_template = '{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for img_name in pinhole_dict:
        # w, h, fx, fy, cx, cy, qvec, t
        params = pinhole_dict[img_name]
        w = params[0]
        h = params[1]
        fx = params[2]
        fy = params[3]
        cx = params[4]
        cy = params[5]
        qvec = params[6:10]
        tvec = params[10:13]

        cam_line = cameras_line_template.format(camera_id="{camera_id}", width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                            tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}", image_name=img_name)
        template[img_name] = (cam_line, img_line)

    # read database
    db = COLMAPDatabase.connect(db_file)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    cameras_txt_lines = []
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        camera_line = template[img_name][0].format(camera_id=img_id)
        cameras_txt_lines.append(camera_line)

        image_line = template[img_name][1].format(image_id=img_id, camera_id=img_id)
        images_txt_lines.append(image_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    fp.close()


def run_point_triangulation(img_dir, db_file, out_dir):
    print('Running point triangulation...')

    # triangulate points
    cmd = ' point_triangulator  --database_path {} \
                                      --image_path {} \
                                      --input_path {} \
                                      --output_path {} \
                                      --Mapper.tri_ignore_two_view_tracks 1'.format(db_file, img_dir, out_dir, out_dir)
    bash_run(cmd)


# this step is optional
def run_global_ba(in_dir, out_dir):
    print('Running global BA...')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    cmd = ' bundle_adjuster --input_path {in_dir} --output_path {out_dir}'.format(in_dir=in_dir, out_dir=out_dir)

    bash_run(cmd)


def prepare_mvs(img_dir, sfm_dir, mvs_dir):
    if not os.path.exists(mvs_dir):
        os.mkdir(mvs_dir)

    images_symlink = os.path.join(mvs_dir, 'images')
    if os.path.exists(images_symlink):
       os.unlink(images_symlink)
    os.symlink(os.path.relpath(img_dir, mvs_dir),
               images_symlink)

    sparse_symlink = os.path.join(mvs_dir, 'sparse')
    if os.path.exists(sparse_symlink):
       os.unlink(sparse_symlink)
    os.symlink(os.path.relpath(sfm_dir, mvs_dir),
               sparse_symlink)

    # prepare stereo directory
    stereo_dir = os.path.join(mvs_dir, 'stereo')
    for subdir in [stereo_dir, 
                   os.path.join(stereo_dir, 'depth_maps'),
                   os.path.join(stereo_dir, 'normal_maps'),
                   os.path.join(stereo_dir, 'consistency_graphs')]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

    # write patch-match.cfg and fusion.cfg
    image_names = sorted(os.listdir(os.path.join(mvs_dir, 'images')))

    with open(os.path.join(stereo_dir, 'patch-match.cfg'), 'w') as fp:
        for img_name in image_names:
            fp.write(img_name + '\n__auto__, 20\n')
            
            # use all images
            # fp.write(img_name + '\n__all__\n')

            # randomly choose 20 images
            # from random import shuffle
            # candi_src_images = [x for x in image_names if x != img_name]
            # shuffle(candi_src_images)
            # max_src_images = 10
            # fp.write(img_name + '\n' + ', '.join(candi_src_images[:max_src_images]) + '\n')

    with open(os.path.join(stereo_dir, 'fusion.cfg'), 'w') as fp:
        for img_name in image_names:
            fp.write(img_name + '\n')


def run_photometric_mvs(mvs_dir, window_radius):
    print('Running photometric MVS...')

    cmd = ' patch_match_stereo --workspace_path {} \
                    --PatchMatchStereo.window_radius {} \
                    --PatchMatchStereo.min_triangulation_angle 3.0 \
                    --PatchMatchStereo.filter 1 \
                    --PatchMatchStereo.geom_consistency 1 \
                    --PatchMatchStereo.gpu_index={} \
                    --PatchMatchStereo.num_samples 15 \
                    --PatchMatchStereo.num_iterations 12'.format(mvs_dir,
                                                                 window_radius, gpu_index)

    bash_run(cmd)


def run_fuse(mvs_dir, out_ply):
    print('Running depth fusion...')

    cmd = ' stereo_fusion --workspace_path {} \
                         --output_path {} \
                         --input_type geometric'.format(mvs_dir, out_ply)
    bash_run(cmd)


def run_possion_mesher(in_ply, out_ply, trim):
    print('Running possion mesher...')

    cmd = ' poisson_mesher \
            --input_path {} \
            --output_path {} \
            --PoissonMeshing.trim {}'.format(in_ply, out_ply, trim)

    bash_run(cmd)


def main(img_dir, pinhole_dict_file, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    db_file = os.path.join(out_dir, 'database.db')
    run_sift_matching(img_dir, db_file)

    sfm_dir = os.path.join(out_dir, 'sfm')
    create_init_files(pinhole_dict_file, db_file, sfm_dir)
    run_point_triangulation(img_dir, db_file, sfm_dir)

    # # optional
    # run_global_ba(sfm_dir, sfm_dir)

    mvs_dir = os.path.join(out_dir, 'mvs')
    prepare_mvs(img_dir, sfm_dir, mvs_dir)
    run_photometric_mvs(mvs_dir, window_radius=5)

    out_ply = os.path.join(out_dir, 'fused.ply')
    run_fuse(mvs_dir, out_ply)

    out_mesh_ply = os.path.join(out_dir, 'meshed_trim_3.ply')
    run_possion_mesher(out_ply, out_mesh_ply, trim=3)


def convert_cam_dict_to_pinhole_dict(cam_dict_file, pinhole_dict_file, img_dir):
    print('Writing pinhole_dict to: ', pinhole_dict_file)

    with open(cam_dict_file) as fp:
        cam_dict = json.load(fp)

    pinhole_dict = {}
    for img_name in cam_dict:
        data_item = cam_dict[img_name]
        if 'img_size' in data_item:
            w, h = data_item['img_size']
        else:
            im = imageio.imread(os.path.join(img_dir, img_name))
            h, w = im.shape[:2]

        K = np.array(data_item['K']).reshape((4, 4))
        W2C = np.array(data_item['W2C']).reshape((4, 4))

        # params
        fx = K[0, 0]
        fy = K[1, 1]
        assert(np.isclose(K[0, 1], 0.))
        cx = K[0, 2]
        cy = K[1, 2]

        print(img_name)
        R = W2C[:3, :3]
        print(R)
        u, s_old, vh = np.linalg.svd(R, full_matrices=False)
        s = np.round(s_old)
        print('s: {} ---> {}'.format(s_old, s))
        R = np.dot(u * s, vh)

        qvec = Quaternion(matrix=R)
        tvec = W2C[:3, 3]

        params = [w, h, fx, fy, cx, cy, 
                  qvec[0], qvec[1], qvec[2], qvec[3],
                  tvec[0], tvec[1], tvec[2]]
        pinhole_dict[img_name] = params

    with open(pinhole_dict_file, 'w') as fp:
        json.dump(pinhole_dict, fp, indent=2, sort_keys=True)


if __name__ == '__main__':
    img_dir = ''
    cam_dict_file = ''
    out_dir = ''

    os.makedirs(out_dir, exist_ok=True)
    pinhole_dict_file = os.path.join(out_dir, 'pinhole_dict.json')
    convert_cam_dict_to_pinhole_dict(cam_dict_file, pinhole_dict_file, img_dir)

    main(img_dir, pinhole_dict_file, out_dir)
