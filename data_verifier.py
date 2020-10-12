import cv2
import numpy as np


## pip install opencv-python=3.4.2.17 opencv-contrib-python==3.4.2.17


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def two_view_geometry(intrinsics1, extrinsics1, intrinsics2, extrinsics2):
    '''
    :param intrinsics1: 4 by 4 matrix
    :param extrinsics1: 4 by 4 W2C matrix
    :param intrinsics2: 4 by 4 matrix
    :param extrinsics2: 4 by 4 W2C matrix
    :return:
    '''
    relative_pose = extrinsics2.dot(np.linalg.inv(extrinsics1))
    R = relative_pose[:3, :3]
    T = relative_pose[:3, 3]
    tx = skew(T)
    E = np.dot(tx, R)
    F = np.linalg.inv(intrinsics2[:3, :3]).T.dot(E).dot(np.linalg.inv(intrinsics1[:3, :3]))

    return E, F, relative_pose


def drawpointslines(img1, img2, lines1, pts2, color):
    '''
    draw corresponding epilines on img1 for the points in img2
    '''

    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt2, cl in zip(lines1, pts2, color):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cl = tuple(cl.tolist())
        img1 = cv2.line(img1, (x0,y0), (x1,y1), cl, 1)
        img2 = cv2.circle(img2, tuple(pt2), 5, cl, -1)
    return img1, img2


def epipolar(coord1, F, img1, img2):
    # compute epipole
    pts1 = coord1.astype(int).T
    color = np.random.randint(0, high=255, size=(len(pts1), 3))
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3, img4 = drawpointslines(img2,img1,lines2,pts1,color)
    ## print(img3.shape)
    ## print(np.concatenate((img4, img3)).shape)
    ## cv2.imwrite('vis.png', np.concatenate((img4, img3), axis=1))

    return np.concatenate((img4, img3), axis=1)


def verify_data(img1, img2, intrinsics1, extrinsics1, intrinsics2, extrinsics2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    E, F, relative_pose = two_view_geometry(intrinsics1, extrinsics1,
                                            intrinsics2, extrinsics2)

    # sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
    # kp1 = sift.detect(img1, mask=None)
    # coord1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1]).T

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp1 = orb.detect(img1, None)
    coord1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1[:20]]).T
    return epipolar(coord1, F, img1, img2)


if __name__ == '__main__':
    from data_loader import load_data
    from run_nerf import config_parser
    from nerf_sample_ray import parse_camera
    import os

    parser = config_parser()
    args = parser.parse_args()
    print(args)

    data = load_data(args.datadir, args.scene, testskip=1)

    all_imgs = data['images']
    all_cameras = data['cameras']
    all_intrinsics = []
    all_extrinsics = []     # W2C
    for i in range(all_cameras.shape[0]):
        W, H, intrinsics, extrinsics = parse_camera(all_cameras[i])
        all_intrinsics.append(intrinsics)
        all_extrinsics.append(np.linalg.inv(extrinsics))

    #### arbitrarily select 10 pairs of images to verify pose
    out_dir = os.path.join(args.basedir, args.expname, 'data_verify')
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    def calc_angles(c2w_1, c2w_2):
        c1 = c2w_1[:3, 3:4]
        c2 = c2w_2[:3, 3:4]

        c1 = c1 / np.linalg.norm(c1)
        c2 = c2 / np.linalg.norm(c2)
        return np.rad2deg(np.arccos(np.dot(c1.T, c2)))

    images_verify = []
    for i in range(10):
        while True:
            idx1, idx2 = np.random.choice(len(all_imgs), (2,), replace=False)

            angle = calc_angles(np.linalg.inv(all_extrinsics[idx1]), 
                                np.linalg.inv(all_extrinsics[idx2]))
            if angle > 5. and angle < 10.:
                break

        im = verify_data(np.uint8(all_imgs[idx1]*255.), np.uint8(all_imgs[idx2]*255.),
                         all_intrinsics[idx1],  all_extrinsics[idx1],
                         all_intrinsics[idx2], all_extrinsics[idx2])
        cv2.imwrite(os.path.join(out_dir, '{:03d}.png'.format(i)), im)
