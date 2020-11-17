import os
import subprocess
from extract_sfm import extract_all_to_dir
from normalize_cam_dict import normalize_cam_dict

#########################################################################
# Note: configure the colmap_bin to the colmap executable on your machine
#########################################################################

def bash_run(cmd):
    colmap_bin = '/home/zhangka2/code/colmap/build/__install__/bin/colmap'
    cmd = colmap_bin + ' ' + cmd
    print('\nRunning cmd: ', cmd)

    subprocess.check_call(['/bin/bash', '-c', cmd])


gpu_index = '-1'


def run_sift_matching(img_dir, db_file, remove_exist=False):
    print('Running sift matching...')

    if remove_exist and os.path.exists(db_file):
        os.remove(db_file) # otherwise colmap will skip sift matching

    # feature extraction
    # if there's no attached display, cannot use feature extractor with GPU
    cmd = ' feature_extractor --database_path {} \
                                    --image_path {} \
                                    --ImageReader.single_camera 1 \
                                    --ImageReader.camera_model SIMPLE_RADIAL \
                                    --SiftExtraction.max_image_size 5000  \
                                    --SiftExtraction.estimate_affine_shape 0 \
                                    --SiftExtraction.domain_size_pooling 1 \
                                    --SiftExtraction.use_gpu 1 \
                                    --SiftExtraction.max_num_features 16384 \
                                    --SiftExtraction.gpu_index {}'.format(db_file, img_dir, gpu_index)
    bash_run(cmd)

    # feature matching
    cmd = ' exhaustive_matcher --database_path {} \
                                     --SiftMatching.guided_matching 1 \
                                     --SiftMatching.use_gpu 1 \
                                     --SiftMatching.max_num_matches 65536 \
                                     --SiftMatching.max_error 3 \
                                     --SiftMatching.gpu_index {}'.format(db_file, gpu_index)

    bash_run(cmd)


def run_sfm(img_dir, db_file, out_dir):
    print('Running SfM...')

    cmd = ' mapper \
            --database_path {} \
            --image_path {} \
            --output_path {} \
            --Mapper.tri_min_angle 3.0 \
            --Mapper.filter_min_tri_angle 3.0'.format(db_file, img_dir, out_dir)
 
    bash_run(cmd)


def prepare_mvs(img_dir, sparse_dir, mvs_dir):
    print('Preparing for MVS...')

    cmd = ' image_undistorter \
            --image_path {} \
            --input_path {} \
            --output_path {} \
            --output_type COLMAP \
            --max_image_size 2000'.format(img_dir, sparse_dir, mvs_dir)

    bash_run(cmd)


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


def main(img_dir, out_dir, run_mvs=False):
    os.makedirs(out_dir, exist_ok=True)

    #### run sfm
    sfm_dir = os.path.join(out_dir, 'sfm')
    os.makedirs(sfm_dir, exist_ok=True)

    img_dir_link = os.path.join(sfm_dir, 'images')
    if os.path.exists(img_dir_link):
        os.remove(img_dir_link)
    os.symlink(img_dir, img_dir_link)

    db_file = os.path.join(sfm_dir, 'database.db')
    run_sift_matching(img_dir, db_file, remove_exist=False)
    sparse_dir = os.path.join(sfm_dir, 'sparse')
    os.makedirs(sparse_dir, exist_ok=True)
    run_sfm(img_dir, db_file, sparse_dir)

    # undistort images
    mvs_dir = os.path.join(out_dir, 'mvs')
    os.makedirs(mvs_dir, exist_ok=True)
    prepare_mvs(img_dir, sparse_dir, mvs_dir)

    # extract camera parameters and undistorted images
    os.makedirs(os.path.join(out_dir, 'posed_images'), exist_ok=True)
    extract_all_to_dir(os.path.join(mvs_dir, 'sparse'), os.path.join(out_dir, 'posed_images'))
    undistorted_img_dir = os.path.join(mvs_dir, 'images')
    posed_img_dir_link = os.path.join(out_dir, 'posed_images/images')
    if os.path.exists(posed_img_dir_link):
        os.remove(posed_img_dir_link)
    os.symlink(undistorted_img_dir, posed_img_dir_link)
    # normalize average camera center to origin, and put all cameras inside the unit sphere
    normalize_cam_dict(os.path.join(out_dir, 'posed_images/kai_cameras.json'),
                       os.path.join(out_dir, 'posed_images/kai_cameras_normalized.json'))

    if run_mvs:
        # run mvs
        run_photometric_mvs(mvs_dir, window_radius=7)

        out_ply = os.path.join(out_dir, 'mvs/fused.ply')
        run_fuse(mvs_dir, out_ply)

        out_mesh_ply = os.path.join(out_dir, 'mvs/meshed_trim_3.ply')
        run_possion_mesher(out_ply, out_mesh_ply, trim=3)


if __name__ == '__main__':
    ### note: this script is intended for the case where all images are taken by the same camera, i.e., intrinisics are shared.
    
    img_dir = ''
    out_dir = ''
    run_mvs = False
    main(img_dir, out_dir, run_mvs=run_mvs)

