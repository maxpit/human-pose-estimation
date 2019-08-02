###############
img = result["image"]
print("IMAGE SHAPE: ", img.shape)

joints = result["joints"]
verts = result["verts"]
cams = result["Cams"]
#rep_joints = reproject_vertices(joints, cams,
 #                  img.shape[1:3])
#verts_calc = tf.multiply(tf.add(verts_reprojected,
#                                tf.ones_like(verts_reprojected)), 0.5)

verts = (verts[0] + 1)*0.5
verts = (verts * img.shape[1:3]).astype(int)

joints = (joints[0] + 1) * 0.5
joints = (joints * img.shape[1:3]).astype(int)

#print("JOINTS SHAPE: ", joints.shape)
#import matplotlib as mpl
import matplotlib.pyplot as plt
for i in range(4):
    plt.imshow(img[0])
    plt.plot(verts[:, 0], verts[:, 1], 'ro')
    plt.plot(joints[i+15,0], joints[i+15,1], 'bo')
    plt.show()
###############
#   0: right foot; 1: right knee; 2: right hip; 3: left hip; 4: left knee; 5: left foot; 6: right wrist
#   7: right elbow; 8: right shoulder; 9: left shoulder; 10: left elbow; 11: left wrist; 12: neck;
#   13-19: head/face