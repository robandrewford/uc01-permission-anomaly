import os
import shutil
import sys
import re

def create_use_case(uc_id, name):
    # Normalize inputs
    # uc_id: e.g. "uc02" or "UC-02" -> "uc02"
    uc_id_clean = uc_id.lower().replace("-", "").replace("_", "")
    if not uc_id_clean.startswith("uc"):
        uc_id_clean = "uc" + uc_id_clean
        
    # name: "Least Privilege Scoring" -> "least-privilege-scoring"
    name_slug = name.lower().replace(" ", "-")
    name_pascal = name.title().replace(" ", "") # LeastPrivilegeScoring
    
    dir_name = f"{uc_id_clean}-{name_slug}"
    base_path = os.getcwd()
    target_dir = os.path.join(base_path, dir_name)
    
    print(f"Scaffolding Use Case: {dir_name}...")
    
    if os.path.exists(target_dir):
        print(f"Directory {target_dir} already exists.")
        return

    # 1. Create Directories
    os.makedirs(os.path.join(target_dir, "notebooks"))
    os.makedirs(os.path.join(target_dir, "scripts"))
    print(f" - Created directories.")

    # 2. Create Placeholder Notebook
    init_notebook = os.path.join(target_dir, "notebooks", "00_init.py")
    with open(init_notebook, "w") as f:
        f.write(f"# {uc_id_clean.upper()} - {name} - Initialization\n")
        f.write("print('Hello from Fabric!')\n")
    print(f" - Created sample notebook: 00_init.py")

    # 3. Copy & Template Deployment Script
    # Source is hardcoded to known good UC01 script for now, or we could have a dedicated template.
    # We'll rely on the one we just fixed in uc01.
    source_deploy_script = os.path.join(base_path, "uc01-permission-anomaly", "scripts", "deploy_to_fabric.py")
    target_deploy_script = os.path.join(target_dir, "scripts", "deploy_to_fabric.py")
    
    if os.path.exists(source_deploy_script):
        with open(source_deploy_script, "r") as f:
            content = f.read()
            
        # Replace UC01 specific strings with new UC ID
        # Note: The script uses "UC01" prefix for naming resources.
        content = content.replace("UC01", uc_id_clean.upper())
        content = content.replace("PermissionAnomaly", name_pascal) # Lakehouse name part
        
        with open(target_deploy_script, "w") as f:
            f.write(content)
        print(f" - Generated deployment script from UC01 template.")
    else:
        print(f" [WARN] Could not find template at {source_deploy_script}")

    # 4. Create .env.template
    env_template = os.path.join(target_dir, ".env.template")
    with open(env_template, "w") as f:
        f.write("# Azure Service Principal Secret\n")
        f.write("AZURE_CLIENT_SECRET=\n")
    print(f" - Created .env.template")

    # 5. Create pyproject.toml
    pyproject = os.path.join(target_dir, "pyproject.toml")
    with open(pyproject, "w") as f:
        f.write(f'[project]\nname = "{dir_name}"\nversion = "0.1.0"\n')
        f.write('dependencies = [\n    "azure-identity>=1.15.0",\n    "requests>=2.31.0",\n    "python-dotenv>=1.0.0",\n]\n')
    print(f" - Created pyproject.toml")
    
    print(f"\n[SUCCESS] {dir_name} checks out.")
    print(f"Next Steps:")
    print(f"1. cd {dir_name}")
    print(f"2. cp .env.template .env (and add secret)")
    print(f"3. Add notebooks to notebooks/")
    print(f"4. uv run python scripts/deploy_to_fabric.py")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/new_usecase.py <ID> <Name>")
        print("Example: python scripts/new_usecase.py UC-02 'Least Privilege Scoring'")
    else:
        create_use_case(sys.argv[1], sys.argv[2])
