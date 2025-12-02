"""
Create comprehensive test data for CAD generation
Generates test CAD scripts covering various complexity levels
"""

import os
import json

def create_test_cad_scripts():
    """Create test CAD scripts directory with various examples"""
    
    scripts_dir = "test_cad_scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Dictionary of test scripts
    test_scripts = {
        
        # BASIC PRIMITIVES
        "01_simple_cube.py": '''"""Simple Cube - Basic primitive"""
import trimesh
mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
''',
        
        "02_sphere.py": '''"""Sphere - Icosphere subdivision"""
import trimesh
mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.5)
''',
        
        "03_cylinder.py": '''"""Cylinder - Basic cylindrical shape"""
import trimesh
mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=32)
''',
        
        "04_cone.py": '''"""Cone - Conical shape"""
import trimesh
mesh = trimesh.creation.cone(radius=1.5, height=3.0, sections=32)
''',
        
        # INTERMEDIATE SHAPES
        "05_torus.py": '''"""Torus - Donut shape"""
import trimesh
import numpy as np

major_radius = 2.0
minor_radius = 0.5
major_sections = 48
minor_sections = 24

vertices = []
for i in range(major_sections):
    theta = 2 * np.pi * i / major_sections
    for j in range(minor_sections):
        phi = 2 * np.pi * j / minor_sections
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
        z = minor_radius * np.sin(phi)
        vertices.append([x, y, z])

vertices = np.array(vertices)

faces = []
for i in range(major_sections):
    for j in range(minor_sections):
        v1 = i * minor_sections + j
        v2 = i * minor_sections + (j + 1) % minor_sections
        v3 = ((i + 1) % major_sections) * minor_sections + (j + 1) % minor_sections
        v4 = ((i + 1) % major_sections) * minor_sections + j
        faces.extend([[v1, v2, v3], [v1, v3, v4]])

mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
''',
        
        "06_capsule.py": '''"""Capsule - Cylinder with hemispherical ends"""
import trimesh
mesh = trimesh.creation.capsule(radius=0.5, height=2.0)
''',
        
        "07_octahedron.py": '''"""Octahedron - Platonic solid"""
import trimesh
import numpy as np

vertices = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
]) * 1.5

faces = np.array([
    [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
    [1, 2, 5], [1, 5, 3], [1, 3, 4], [1, 4, 2]
])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
''',
        
        "08_uv_sphere.py": '''"""UV Sphere - Parametric sphere"""
import trimesh
mesh = trimesh.creation.uv_sphere(radius=1.5, count=[32, 32])
''',
        
        # COMPLEX SHAPES
        "09_multi_primitive.py": '''"""Multi-Primitive Composition"""
import trimesh
import numpy as np

# Create base
base = trimesh.creation.cylinder(radius=2.0, height=0.5, sections=32)

# Create middle sphere
sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.5)
sphere.apply_translation([0, 0, 1.5])

# Create top cone
cone = trimesh.creation.cone(radius=1.0, height=2.0, sections=32)
cone.apply_translation([0, 0, 3.0])

# Combine
mesh = trimesh.util.concatenate([base, sphere, cone])
''',
        
        "10_star_extrusion.py": '''"""Star Shape - 2D to 3D extrusion"""
import trimesh
import numpy as np

# Create star points
num_points = 10
inner_radius = 0.5
outer_radius = 1.5
height = 0.3

vertices_2d = []
for i in range(num_points):
    angle = 2 * np.pi * i / num_points
    if i % 2 == 0:
        r = outer_radius
    else:
        r = inner_radius
    vertices_2d.append([r * np.cos(angle), r * np.sin(angle)])

# Extrude to 3D
vertices = []
for v in vertices_2d:
    vertices.append([v[0], v[1], 0])
    vertices.append([v[0], v[1], height])

vertices = np.array(vertices)

# Create faces
faces = []
n = len(vertices_2d)
# Bottom face
bottom = list(range(0, 2*n, 2))
for i in range(len(bottom)):
    faces.append([bottom[i], bottom[(i+1)%len(bottom)], bottom[(i+2)%len(bottom)]])

# Top face  
top = list(range(1, 2*n, 2))
for i in range(len(top)):
    faces.append([top[i], top[(i+2)%len(top)], top[(i+1)%len(top)]])

# Side faces
for i in range(n):
    v1 = 2*i
    v2 = 2*i + 1
    v3 = 2*((i+1)%n) + 1
    v4 = 2*((i+1)%n)
    faces.extend([[v1, v2, v3], [v1, v3, v4]])

mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
''',
        
        "11_helical_spring.py": '''"""Helical Spring"""
import trimesh
import numpy as np

# Spring parameters
radius = 1.0
coil_radius = 0.2
num_coils = 5
points_per_coil = 32

# Generate helix path
t = np.linspace(0, num_coils * 2 * np.pi, num_coils * points_per_coil)
x = radius * np.cos(t)
y = radius * np.sin(t)
z = t / (2 * np.pi) * 0.5

# Create tube along helix
path = np.column_stack([x, y, z])

# Simple tube approximation
vertices = []
faces = []
tube_sections = 8

for i, point in enumerate(path[:-1]):
    direction = path[i+1] - point
    direction = direction / np.linalg.norm(direction)
    
    # Create perpendicular vectors
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, [0, 0, 1])
    else:
        perp1 = np.cross(direction, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    # Create circle of vertices
    for j in range(tube_sections):
        angle = 2 * np.pi * j / tube_sections
        offset = coil_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(point + offset)

vertices = np.array(vertices)

# Create faces
for i in range(len(path) - 2):
    for j in range(tube_sections):
        v1 = i * tube_sections + j
        v2 = i * tube_sections + (j + 1) % tube_sections
        v3 = (i + 1) * tube_sections + (j + 1) % tube_sections
        v4 = (i + 1) * tube_sections + j
        faces.extend([[v1, v2, v3], [v1, v3, v4]])

mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
''',
        
        "12_dodecahedron.py": '''"""Dodecahedron - 12-sided Platonic solid"""
import trimesh
import numpy as np

phi = (1 + np.sqrt(5)) / 2

vertices = np.array([
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
    [1/phi, 0, phi], [-1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, -phi],
    [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
])

# Simplified face generation for dodecahedron
faces = []
# This is a simplified version, proper dodecahedron faces would require more work
for i in range(len(vertices)-2):
    for j in range(i+1, len(vertices)-1):
        for k in range(j+1, len(vertices)):
            # Check if three vertices form a reasonable face
            v1, v2, v3 = vertices[i], vertices[j], vertices[k]
            if np.linalg.norm(v1-v2) < 2.5 and np.linalg.norm(v2-v3) < 2.5 and np.linalg.norm(v3-v1) < 2.5:
                faces.append([i, j, k])

mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces[:40]))  # Limit faces
'''
    }
    
    # Write all scripts
    print(f"Creating {len(test_scripts)} test CAD scripts...")
    for filename, content in test_scripts.items():
        filepath = os.path.join(scripts_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Created: {filename}")
    
    print(f"\n✓ Successfully created {len(test_scripts)} test scripts in {scripts_dir}/")
    
    return len(test_scripts)


def create_test_data_json():
    """Create JSON files for training/validation prompts"""
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Prompts for CAD generation
    prompts_data = {
        "train": [
            {"prompt": "Generate a cube with side length 2.0", "description": "Simple cube"},
            {"prompt": "Create a sphere with radius 1.5", "description": "Sphere"},
            {"prompt": "Generate a cylinder with radius 1.0 and height 3.0", "description": "Cylinder"},
            {"prompt": "Create a cone with base radius 1.5 and height 2.5", "description": "Cone"},
            {"prompt": "Generate a torus with major radius 2.0 and minor radius 0.5", "description": "Torus"},
            {"prompt": "Create a capsule with radius 0.5 and height 2.0", "description": "Capsule"},
            {"prompt": "Generate an octahedron with side length 1.5", "description": "Octahedron"},
            {"prompt": "Create a rectangular box with dimensions 2x3x4", "description": "Box"},
            {"prompt": "Generate a UV sphere with radius 1.5", "description": "UV Sphere"},
            {"prompt": "Create a pyramid with base side 2.0 and height 3.0", "description": "Pyramid"},
            {"prompt": "Generate a hexagonal prism", "description": "Hexagonal prism"},
            {"prompt": "Create a star-shaped extrusion", "description": "Star shape"},
            {"prompt": "Generate a helical spring", "description": "Spring"},
            {"prompt": "Create a dodecahedron", "description": "Dodecahedron"},
            {"prompt": "Generate a composite shape with multiple primitives", "description": "Multi-primitive"},
        ] * 5,  # Repeat for more training data
        
        "val": [
            {"prompt": "Create a rounded box", "description": "Rounded box"},
            {"prompt": "Generate a gear with 12 teeth", "description": "Gear"},
            {"prompt": "Create a bottle shape using lofting", "description": "Bottle"},
            {"prompt": "Generate a table with 4 legs", "description": "Table"},
        ] * 3
    }
    
    prompts_path = os.path.join(data_dir, "prompts.json")
    with open(prompts_path, 'w') as f:
        json.dump(prompts_data, f, indent=2)
    print(f"✓ Created prompts data: {prompts_path}")
    
    # Code examples for supervised pre-training
    code_data = {
        "train": [
            {
                "prompt": "Generate a cube",
                "code": """import trimesh
mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])"""
            },
            {
                "prompt": "Create a sphere",
                "code": """import trimesh
mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.5)"""
            },
            {
                "prompt": "Generate a cylinder",
                "code": """import trimesh
mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=32)"""
            },
        ] * 10
    }
    
    code_path = os.path.join(data_dir, "code_examples.json")
    with open(code_path, 'w') as f:
        json.dump(code_data, f, indent=2)
    print(f"✓ Created code examples: {code_path}")


def main():
    """Main function to create all test data"""
    print("="*80)
    print("CREATING TEST DATA FOR CAD RL TRAINING")
    print("="*80)
    
    # Create test CAD scripts
    num_scripts = create_test_cad_scripts()
    
    print("\n" + "="*80)
    
    # Create JSON data files
    create_test_data_json()
    
    print("\n" + "="*80)
    print("TEST DATA CREATION COMPLETE")
    print("="*80)
    print(f"✓ Created {num_scripts} test CAD scripts")
    print(f"✓ Created training prompts and code examples")
    print("\nDirectories created:")
    print("  📁 test_cad_scripts/ - Test CAD generation scripts")
    print("  📁 data/ - Training and validation data")
    print("\nYou can now run:")
    print("  • python run_all_mesh_tests.py - Test mesh conversions")
    print("  • python test.py - Test all components")
    print("  • python train.py - Start training")


if __name__ == '__main__':
    main()