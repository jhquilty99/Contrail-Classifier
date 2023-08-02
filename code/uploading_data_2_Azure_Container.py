# Upload data to Azure cloud

import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

connect_str = "DefaultEndpointsProtocol=https;AccountName=contrailclassi9184983434;AccountKey=5PDmYTyBayyV8/18iL75zYTmGnjVFllFe+TM7Y6kRaS0qezJOoBfCL7u+GEfI8oT56cimhn8XZ8h+AStEMwZ3A==;EndpointSuffix=core.windows.net"
container_name = "contrail-classifier-data"

# Creates the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

directory = "..\data"

# Walks through all the subfolders in the directory to upload data to Azure and preserves file structure
for root, dirs, files in os.walk(directory):
    for file in files:
        file_path = os.path.join(root, file)
        
        blob_name = file_path.replace(directory, "")
        
        # Creates a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container_name, blob_name)

        # Uploads the file
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data)

#  Creates a datastore in your Azure Machine Learning workspace that points to the Blob Storage container.

from azure.ai.ml.entities import AzureBlobDatastore
from azure.ai.ml import MLClient

ml_client = MLClient.from_config()

store = AzureBlobDatastore(
 name="contrail_datastore",
 description="stores images from contrail classifer",
 account_name="contrailclassi9184983434",
 container_name="contrail-classifier-data "
)

ml_client.create_or_update(store)