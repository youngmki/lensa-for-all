import os
import yaml
import boto3
from utils.logger import logger
from utils.misc import (
    create_bucket_if_not_exists,
    delete_files_in_s3,
    dump_json,
    upload_dir_to_s3,
)


if __name__ == "__main__":
    with open(os.path.join("config", "config.yaml"), encoding="utf-8") as file_path:
        config = yaml.safe_load(file_path)

    profile_name = config["environment"]["iam_profile_name"]
    region_name = config["environment"]["region_name"]
    dataset_dir = config["environment"]["ebs_dataset_dir"]
    bucket = config["environment"]["s3_bucket"]
    base_prefix = config["environment"]["s3_base_prefix"]
    dataset_prefix = config["environment"]["s3_dataset_prefix"]

    subject_name = config["input"]["subject_name"].lower()
    class_name = config["input"]["class_name"].lower()

    use_jumpstart = config["model"]["use_jumpstart"]

    # Uploading Files to S3

    boto_session = boto3.Session(profile_name=profile_name, region_name=region_name)

    dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")
    if use_jumpstart:
        dataset_info = {
            "instance_prompt": f"a photo of {subject_name} {class_name}",
            "class_prompt": f"a photo of {class_name}",
        }
        dump_json(dataset_info_path, dataset_info)

    elif os.path.exists(dataset_info_path):
        os.remove(dataset_info_path)

    bucket = (
        create_bucket_if_not_exists(boto_session, region_name, logger=logger)
        if bucket is None
        else bucket
    )

    delete_files_in_s3(
        boto_session, bucket, f"{base_prefix}/{dataset_prefix}", logger=logger
    )

    upload_dir_to_s3(
        boto_session,
        dataset_dir,
        bucket,
        f"{base_prefix}/{dataset_prefix}",
        file_ext_to_excl=["DS_Store"],
        logger=logger,
    )

    logger.info("The 'Uploading images' job finished successfully.")
