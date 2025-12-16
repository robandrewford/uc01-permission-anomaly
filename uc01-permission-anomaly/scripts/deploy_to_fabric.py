import os
import json
import base64
import requests
import glob
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.core.exceptions import ClientAuthenticationError

# Load environment variables from .env file
load_dotenv()

# Configuration - Populated from User Inputs
TENANT_ID = "f81303a9-1e61-4c79-8594-04264b2b02e4"
CLIENT_ID = "00219831-e4ac-49e5-8190-cfd52c54799f"
WORKSPACE_ID = "c42136d7-2d61-4539-be97-935aafec18aa"

# Environment Variable for Secret
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

if not CLIENT_SECRET:
    raise ValueError("Please set the AZURE_CLIENT_SECRET environment variable or add it to a .env file.")

class FabricDeployer:
    def __init__(self):
        print(f"Authenticating as Service Principal (Client ID: {CLIENT_ID})...")
        try:
            self.credential = ClientSecretCredential(
                tenant_id=TENANT_ID,
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET
            )
            self.token = self.credential.get_token("https://api.fabric.microsoft.com/.default").token
            print("Authentication successful.")
        except ClientAuthenticationError as e:
            print(f"Authentication failed: {e}")
            raise

        self.base_url = f"https://api.fabric.microsoft.com/v1/workspaces/{WORKSPACE_ID}"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def _get_item_id_by_name(self, name, item_type):
        response = requests.get(f"{self.base_url}/items?type={item_type}", headers=self.headers)
        if response.status_code == 200:
            items = response.json().get('value', [])
            for item in items:
                # Case-insensitive match to handle "Generate" vs "generate"
                if item['displayName'].lower() == name.lower():
                    return item['id']
        return None

    def create_lakehouse(self, name):
        print(f"Checking Lakehouse: {name}...")
        existing_id = self._get_item_id_by_name(name, "Lakehouse")
        if existing_id:
            print(f" [INFO] Lakehouse '{name}' already exists (ID: {existing_id})")
            return existing_id

        # Create if not exists
        print(f"Creating Lakehouse: {name}...")
        payload = {
            "displayName": name,
            "type": "Lakehouse"
        }
        
        response = requests.post(f"{self.base_url}/items", json=payload, headers=self.headers)
        if response.status_code == 201:
            item_id = response.json()['id']
            print(f" [SUCCESS] Created Lakehouse '{name}' (ID: {item_id})")
            return item_id
        else:
            print(f" [ERROR] Lakehouse creation failed: {response.status_code} - {response.text}")
            return None

    def upload_notebook(self, file_path, name):
        print(f"Processing Notebook: {name}...")
        
        # Check if exists first (Case insensitive check via updated _get_item_id_by_name)
        existing_id = self._get_item_id_by_name(name, "Notebook")
        
        if not os.path.exists(file_path):
            print(f" [ERROR] File not found: {file_path}")
            return existing_id

        # ... (File Reading Same) ...
        script_content = ""
        with open(file_path, 'r') as f:
            script_content = f.read()

        # ... (Nb Struct Same) ...
        nb_struct = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": script_content.splitlines(keepends=True)
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }
        payload_b64 = base64.b64encode(json.dumps(nb_struct).encode()).decode()
        
        notebook_payload = {
            "displayName": name,
            "type": "Notebook",
            "definition": {
                "format": "ipynb",
                "parts": [
                    {
                        "path": "notebook-content.ipynb",
                        "payload": payload_b64,
                        "payloadType": "InlineBase64"
                    }
                ]
            }
        }

        if existing_id:
            print(f" [INFO] Notebook '{name}' exists (ID: {existing_id}). Updating definition...")
            # Wrap definition in "definition" key as required by API
            update_payload = {"definition": notebook_payload['definition']}
            response = requests.post(f"{self.base_url}/items/{existing_id}/updateDefinition", json=update_payload, headers=self.headers)
            if response.status_code in [200, 202]:
                print(f" [SUCCESS] Updated Notebook '{name}'")
                return existing_id
            else:
                print(f" [WARN] Failed to update notebook: {response.status_code} - {response.text}")
                return existing_id
        else:
            print(f"Creating Notebook: {name}...")
            response = requests.post(f"{self.base_url}/items", json=notebook_payload, headers=self.headers)
            if response.status_code == 201:
                item_id = response.json()['id']
                print(f" [SUCCESS] Created Notebook '{name}' (ID: {item_id})")
                return item_id
            elif response.status_code == 202:
                print(f" [SUCCESS] Created Notebook '{name}' (Async)")
                # For async, try to fetch ID again with retries
                import time
                for i in range(10):
                    time.sleep(2)
                    nb_id = self._get_item_id_by_name(name, "Notebook")
                    if nb_id: 
                        return nb_id
                print(f" [WARN] Timed out waiting for Notebook '{name}' ID.")
                return None 
            else:
                print(f" [ERROR] Failed to upload notebook: {response.status_code} - {response.text}")
                return None

    def create_pipeline(self, pipeline_name, notebook_ids_map):
        print(f"Creating Pipeline: {pipeline_name}...")
        
        # Sort notebooks by name to determine order
        sorted_names = sorted(notebook_ids_map.keys())
        
        activities = []
        previous_activity = None
        
        for name in sorted_names:
            nb_id = notebook_ids_map[name]
            if not nb_id: 
                continue

            activity_name = f"Run_{name}"
            # Clean activity name (remove spaces as they might cause issues in dependsOn references, although allowed)
            activity_name = activity_name.replace(" ", "_").replace("-", "_")

            activity = {
                "name": activity_name,
                "type": "TridentNotebook", 
                "typeProperties": {
                    "notebookId": nb_id,
                    "workspaceId": WORKSPACE_ID,
                    "targetCell": None
                },
                "policy": {
                    "timeout": "0.12:00:00",
                    "retry": 0,
                    "retryIntervalInSeconds": 30,
                    "secureInput": False,
                    "secureOutput": False
                }
            }
            
            if previous_activity:
                activity["dependsOn"] = [
                    {
                        "activity": previous_activity,
                        "dependencyConditions": ["Succeeded"]
                    }
                ]
            
            activities.append(activity)
            previous_activity = activity_name

        pipeline_definition = {
            "properties": {
                "activities": activities
            }
        }
        
        pipeline_payload = {
            "displayName": pipeline_name,
            "type": "DataPipeline",
            "definition": {
                "parts": [
                    {
                        "path": "pipeline-content.json",
                        "payload": base64.b64encode(json.dumps(pipeline_definition).encode()).decode(),
                        "payloadType": "InlineBase64"
                    }
                ]
            }
        }
        
        # Check if exists
        existing_id = self._get_item_id_by_name(pipeline_name, "DataPipeline")
        if existing_id:
             print(f" [INFO] Pipeline '{pipeline_name}' exists (ID: {existing_id}). Updating...")
             # Update definition
             update_payload = {"definition": pipeline_payload['definition']}
             response = requests.post(f"{self.base_url}/items/{existing_id}/updateDefinition", json=update_payload, headers=self.headers)
             if response.status_code in [200, 202]:
                 print(f" [SUCCESS] Updated Pipeline '{pipeline_name}'")
             else:
                 print(f" [ERROR] Update failed: {response.status_code} - {response.text}")
        else:
            response = requests.post(f"{self.base_url}/items", json=pipeline_payload, headers=self.headers)
            if response.status_code == 201:
                print(f" [SUCCESS] Created Pipeline '{pipeline_name}'")
            else:
                print(f" [ERROR] Pipeline creation failed: {response.status_code} - {response.text}")

    def run_pipeline(self, pipeline_name):
        print(f"Triggering Pipeline: {pipeline_name}...")
        pipeline_id = self._get_item_id_by_name(pipeline_name, "DataPipeline")
        
        if not pipeline_id:
            print(f" [ERROR] Pipeline '{pipeline_name}' not found.")
            return

        # Trigger Job
        # Endpoint: POST /workspaces/{workspaceId}/items/{itemId}/jobs/instances?jobType=Pipeline
        job_url = f"{self.base_url}/items/{pipeline_id}/jobs/instances?jobType=Pipeline"
        
        response = requests.post(job_url, headers=self.headers)
        
        if response.status_code in [202, 200]:
            print(f" [SUCCESS] Pipeline execution started!")
            # 202 matches, it might return a location header or body with instanceId
             # e.g. Location: https://api.fabric.microsoft.com/v1/workspaces/.../jobs/instances/GUID
            job_location = response.headers.get("Location")
            if job_location:
                 job_id = job_location.split("/")[-1]
                 print(f" [INFO] Job Instance ID: {job_id}")
                 print(f" [INFO] You can monitor this in the Fabric Monitoring Hub.")
        else:
             print(f" [ERROR] Failed to trigger pipeline: {response.status_code} - {response.text}")

    def deploy(self):
        # 1. Create Lakehouse 
        self.create_lakehouse("UC01_PermissionAnomaly_Lakehouse")

        # 2. Discover and Upload Notebooks
        notebook_files = glob.glob("notebooks/*.py")
        notebook_ids = {}
        
        print(f"Found {len(notebook_files)} notebooks to deploy.")
        
        for file_path in sorted(notebook_files):
            filename = os.path.basename(file_path)
            name_stem = os.path.splitext(filename)[0]
            
            # Use Title Case for nicer display names: "00_generate_synthetic_data" -> "00_Generate_Synthetic_Data"
            # And handling the UC01 prefix logic
            
            display_name_stem = name_stem.replace("_", " ").title().replace(" ", "_")
            # e.g. "00 Generate Synthetic Data" -> "00_Generate_Synthetic_Data"
            
            # Heuristic: If filename started with 00, title() probably kept it.
            
            display_name = f"UC01_{display_name_stem}"
            # Check if double prefix happened
            if name_stem.lower().startswith("uc01"):
                 # File was already uc01_...
                 display_name = display_name_stem
            
            nb_id = self.upload_notebook(file_path, display_name)
            if nb_id:
                notebook_ids[display_name] = nb_id

        # 3. Create Pipeline
        if notebook_ids:
            # We sort based on the keys (the display names), which should still be alphabetically correct 
            # if they started with 00, 01, etc.
            self.create_pipeline("UC01_Automated_Pipeline", notebook_ids)
            
            # 4. Run Pipeline
            self.run_pipeline("UC01_Automated_Pipeline")
        else:
            print("[WARN] No notebooks uploaded, skipping pipeline creation.")

if __name__ == "__main__":
    deployer = FabricDeployer()
    deployer.deploy()
