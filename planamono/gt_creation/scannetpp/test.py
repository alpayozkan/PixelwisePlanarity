import open3d as o3d
import numpy as np

print("Creating offscreen renderer...")
try:
    renderer = o3d.visualization.rendering.OffscreenRenderer(320, 240)
except Exception as e:
    print("❌ Renderer creation failed:", e)
    exit()

# Simple geometry
mesh = o3d.geometry.TriangleMesh.create_box()
mesh.compute_vertex_normals()

mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"

print("Adding geometry...")
try:
    renderer.scene.add_geometry("box", mesh, mat)
except Exception as e:
    print("❌ Adding geometry failed:", e)
    exit()

# Camera intrinsics
fx, fy = 200, 200
cx, cy = 160, 120
intr = o3d.camera.PinholeCameraIntrinsic(320, 240, fx, fy, cx, cy)

# Simple camera pose
c2w = np.eye(4)
w2c = np.linalg.inv(c2w)

print("Setting up camera...")
try:
    renderer.setup_camera(intr, w2c)
except Exception as e:
    print("❌ Camera setup failed:", e)
    exit()

print("Rendering RGB...")
try:
    img = renderer.render_to_image()
    print("✔ GPU rendering works! (RGB shape =)", np.asarray(img).shape)
except Exception as e:
    print("❌ Rendering failed:", e)
    exit()