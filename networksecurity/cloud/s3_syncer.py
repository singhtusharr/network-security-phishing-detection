"""Final step of the Model Pusher: sync the artifacts/ and saved_models/
folders up to an S3 bucket, so the latest trained model and its full run
history are durable and shared (not just sitting on one laptop).

Shells out to the `aws` CLI's `s3 sync` rather than using boto3 directly —
`sync` only uploads changed/new files (diffing against what's already in the
bucket), which `boto3.upload_file` doesn't do for you out of the box.
Requires `aws configure` to have been run with real AWS credentials; if it
hasn't (e.g. no AWS account set up yet), this raises a clear error instead of
silently doing nothing.
"""
import subprocess
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class S3Sync:
    @staticmethod
    def sync_folder_to_s3(folder: str, aws_bucket_url: str) -> None:
        try:
            command = f"aws s3 sync {folder} {aws_bucket_url}"
            logging.info(f"Syncing {folder} -> {aws_bucket_url}")
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def sync_folder_from_s3(folder: str, aws_bucket_url: str) -> None:
        try:
            command = f"aws s3 sync {aws_bucket_url} {folder}"
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
