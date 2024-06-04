import cv2
import trimesh
from utils import viz_3d
from mesh_smoothing import sort_onerings, laplace_beltrami_cotan_MC, implicit_smooth

# load mesh
mesh = trimesh.load('meshes/bunny.obj')

# calculate laplacian matrix
nbrs,edge_vertices = sort_onerings(mesh)
L,M,Minv,C = laplace_beltrami_cotan_MC(mesh,nbrs,edge_vertices)

vertices = []
smooth_v = mesh.vertices
vertices.append(smooth_v)
N = 60 * 3
for i in range(N):
    # perform implicit smoothing
    smooth_v = implicit_smooth(0.00001,M,C,smooth_v)
    vertices.append(smooth_v)

# visualize the smoothing
viz_3d(mesh,vertices,"results",save_image=True)
    
# create a video from the images
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('results/mesh_smoothing.avi', fourcc, 12, (1920, 1080), isColor=True)
dir1 = "results/"
images = []
for i in range(N):
    out.write(cv2.imread(f"{dir1}/{i}.png"))
# reverse 
for i in range(N-1, -1, -1):
    out.write(cv2.imread(f"{dir1}/{i}.png"))
out.release()



    
