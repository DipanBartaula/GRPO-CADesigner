import os
import requests
from tqdm import tqdm


MESH_URL = "https://gitee.com/fred926/whucad-mesh.git"
BREP_URL = "https://gitee.com/fred926/whucad-brep.git"

TARGET_DIR = "data/whucad"

def download_whucad(target_dir=TARGET_DIR):
    os.makedirs(target_dir, exist_ok=True)
    # clone with git so you get full repository content
    if not os.path.isdir(os.path.join(target_dir, "whucad-mesh")):
        print(f"Cloning WHUCAD mesh repository from Gitee …")
        os.system(f"git clone {MESH_URL} {os.path.join(target_dir, 'mesh')}")
    else:
        print("WHUCAD mesh already cloned.")
    if not os.path.isdir(os.path.join(target_dir, "whucad-brep")):
        print(f"Cloning WHUCAD brep repository from Gitee …")
        os.system(f"git clone {BREP_URL} {os.path.join(target_dir, 'brep')}")
    else:
        print("WHUCAD brep already cloned.")

if __name__ == "__main__":
    download_whucad()
