# Here are some references.
# https://github.com/huggingface/notebooks/tree/main/sagemaker
# https://github.com/aws/sagemaker-huggingface-inference-toolkit
# https://aws.amazon.com/blogs/machine-learning/hugging-face-on-amazon-sagemaker-bring-your-own-scripts-and-data/
# https://huggingface.co/docs/accelerate/usage_guides/sagemaker
# https://huggingface.co/docs/diffusers/training/dreambooth
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
# https://techpp.com/2022/10/10/how-to-train-stable-diffusion-ai-dreambooth/
# https://github.com/JoePenna/Dreambooth-Stable-Diffusion

import os
import random
import shutil
from pathlib import Path
from pprint import pformat
import boto3
import sagemaker
import yaml
from huggingface_hub import snapshot_download
from sagemaker import (
    hyperparameters,
    image_uris,
    model_uris,
    PipelineModel,
    script_uris,
)
from sagemaker.estimator import Estimator
from sagemaker.huggingface import HuggingFace, HuggingFaceModel
from sagemaker.model import Model
from sagemaker.utils import name_from_base
from sagemaker.s3 import S3Uploader
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter
from utils.logger import logger
from utils.misc import (
    compress_dir_to_model_tar_gz,
    create_bucket_if_not_exists,
    create_role_if_not_exists,
)


IS_MODEL_ALREADY_UPLOADED = False
HF_MODEL_IDS = [
    "Gustavosta/MagicPrompt-Stable-Diffusion",
    "model-txt2img-stabilityai-stable-diffusion-v2-1-base",
    "runwayml/stable-diffusion-v1-5",
]
TRANSFORMERS_VERSION = "4.26.0"
PYTORCH_VERSION = "1.13.1"
PY_VERSION = "py39"
TRAIN_MODEL_VERSION = "*"
TRAIN_SCOPE = "training"
INFER_SCOPE = "inference"


if __name__ == "__main__":
    with open(os.path.join("config", "config.yaml"), encoding="utf-8") as file_path:
        config = yaml.safe_load(file_path)

    profile_name = config["environment"]["iam_profile_name"]
    region_name = config["environment"]["region_name"]
    role = config["environment"]["iam_role"]
    bucket = config["environment"]["s3_bucket"]
    base_prefix = config["environment"]["s3_base_prefix"]
    dataset_prefix = config["environment"]["s3_dataset_prefix"]
    hf_token = config["environment"]["hf_token"]
    wandb_api_key = config["environment"]["wandb_api_key"]

    subject_name = config["input"]["subject_name"].lower()
    class_name = config["input"]["class_name"].lower()

    use_jumpstart = config["model"]["use_jumpstart"]
    model_data = config["model"]["model_data"]
    with_prior_preservation = config["model"]["with_prior_preservation"]
    train_text_encoder = config["model"]["train_text_encoder"]
    max_steps = config["model"]["max_steps"]
    batch_size = config["model"]["batch_size"]
    learning_rate = config["model"]["learning_rate"]
    tune_params = config["model"]["tune_params"]
    max_tuning_jobs = config["model"]["max_tuning_jobs"]
    train_instance_type = config["model"]["train_instance_type"]
    infer_instance_type = config["model"]["infer_instance_type"]
    endpoint_name = config["model"]["sm_endpoint_name"]

    boto_session = boto3.Session(profile_name=profile_name, region_name=region_name)
    sm_session = sagemaker.session.Session(boto_session=boto_session)
    role = (
        create_role_if_not_exists(boto_session, region_name, logger=logger)
        if role is None
        else role
    )
    bucket = (
        create_bucket_if_not_exists(boto_session, region_name, logger=logger)
        if bucket is None
        else bucket
    )

    model_data = (
        model_data
        if model_data is None
        else f"s3://{bucket}/{base_prefix}/output/{model_data}/output/model.tar.gz"
    )

    train_ds_path = f"s3://{bucket}/{base_prefix}/{dataset_prefix}/"
    output_path = f"s3://{bucket}/{base_prefix}/output"

    # Compressing the GPT2 downloaded from the HuggingFace Hub and uploading it to S3

    train_model_uri = (
        f"s3://{bucket}/{base_prefix}/{HF_MODEL_IDS[0].rsplit('/', maxsplit=1)[-1]}"
    )

    if IS_MODEL_ALREADY_UPLOADED:
        train_model_uri += "/model.tar.gz"

    else:
        snapshot_dir = snapshot_download(
            repo_id=HF_MODEL_IDS[0], use_auth_token=hf_token
        )

        model_dir = Path(f"model-{random.getrandbits(16)}")
        model_dir.mkdir(exist_ok=True)

        shutil.copytree(snapshot_dir, str(model_dir), dirs_exist_ok=True)
        shutil.copytree(
            os.path.join("code", "inference", "gpt") + os.path.sep,
            str(model_dir.joinpath("code")),
        )

        compress_dir_to_model_tar_gz(str(model_dir), logger=logger)
        shutil.rmtree(str(model_dir))

        train_model_uri = S3Uploader.upload(
            local_path="model.tar.gz",
            desired_s3_uri=train_model_uri,
        )

        logger.info("model uploaded to: %s", train_model_uri)

    gpt_model = HuggingFaceModel(
        role=role,
        model_data=train_model_uri,
        transformers_version=TRANSFORMERS_VERSION,
        pytorch_version=PYTORCH_VERSION,
        py_version=PY_VERSION,
        sagemaker_session=sm_session,
    )

    # Fine-tuning the Stable Diffusion model in SageMaker Jumpstart with Dreambooth

    if model_data is None:
        if use_jumpstart:
            train_image_uri = image_uris.retrieve(
                framework=None,
                region=region_name,
                instance_type=train_instance_type,
                image_scope=TRAIN_SCOPE,
                model_id=HF_MODEL_IDS[1],
                model_version=TRAIN_MODEL_VERSION,
            )

            train_source_uri = script_uris.retrieve(
                region=region_name,
                model_id=HF_MODEL_IDS[1],
                model_version=TRAIN_MODEL_VERSION,
                script_scope=TRAIN_SCOPE,
            )

            train_model_uri = model_uris.retrieve(
                region=region_name,
                model_id=HF_MODEL_IDS[1],
                model_version=TRAIN_MODEL_VERSION,
                model_scope=TRAIN_SCOPE,
            )

            params = hyperparameters.retrieve_default(
                model_id=HF_MODEL_IDS[1], model_version=TRAIN_MODEL_VERSION
            )
            params["with_prior_preservation"] = (
                "False"
                if with_prior_preservation is None
                else str(with_prior_preservation)
            )
            params["max_steps"] = "None" if max_steps is None else str(max_steps)
            params["batch_size"] = 1 if batch_size is None else batch_size
            params["learning_rate"] = 2e-06 if learning_rate is None else learning_rate
            params["seed"] = 42

            train_job_name = name_from_base(
                f"{base_prefix}-{HF_MODEL_IDS[1].replace('/', '-').replace('_', '-')}"
            )

            estimator = Estimator(
                image_uri=train_image_uri,
                role=role,
                instance_count=1,
                instance_type=train_instance_type,
                output_path=output_path,
                base_job_name=train_job_name,
                sagemaker_session=sm_session,
                hyperparameters=params,
                model_uri=train_model_uri,
                source_dir=train_source_uri,
                entry_point="transfer_learning.py",
            )

            if tune_params:
                params["compute_fid"] = "True"
                param_ranges = {
                    "max_steps": IntegerParameter(200, 1200, "Linear"),
                    "learning_rate": ContinuousParameter(1e-06, 5e-06, "Linear"),
                }
                max_tuning_jobs = 10 if max_tuning_jobs is None else max_tuning_jobs

                estimator.set_hyperparameters(**params)
                metric_definitions = [
                    {"Name": "fid_score", "Regex": "fid_score=([-+]?\\d\\.?\\d*)"},
                ]
                tuner = HyperparameterTuner(
                    estimator=estimator,
                    objective_metric_name="fid_score",
                    hyperparameter_ranges=param_ranges,
                    metric_definitions=metric_definitions,
                    strategy="Bayesian",
                    objective_type="Minimize",
                    max_jobs=max_tuning_jobs,
                    max_parallel_jobs=1,
                    base_tuning_job_name=train_job_name,
                )
                tuner.fit({"training": train_ds_path}, logs=False)
                params = tuner.best_estimator().hyperparameters()

            else:
                _ = estimator.fit({"training": train_ds_path}, logs=True)

        else:
            params = {
                "pretrained_model_name_or_path": HF_MODEL_IDS[2],
                "instance_prompt": f"'a photo of {subject_name} {class_name}'",
                "seed": 42,
                "train_batch_size": 1 if batch_size is None else batch_size,
                "learning_rate": 2e-06 if learning_rate is None else learning_rate,
                "lr_warmup_steps": 0,
                "max_train_steps": None if max_steps is None else max_steps,
                "num_class_images": 200,
            }

            if with_prior_preservation:
                params["with_prior_preservation"] = ""
                params["class_prompt"] = f"'a photo of {class_name}'"
                params["gradient_accumulation_steps"] = 1

                if train_text_encoder:
                    params["train_text_encoder"] = ""
                    params["use_8bit_adam"] = ""
                    params["gradient_checkpointing"] = ""
                    params["enable_xformers_memory_efficient_attention"] = ""
                    params["set_grads_to_none"] = ""

            if wandb_api_key is not None:
                validation_prompt = f"'a photo of {subject_name} {class_name} with Eiffel Tower in the background'"

                params["report_to"] = "wandb"
                params["validation_prompt"] = validation_prompt
                params["wandb_api_key"] = wandb_api_key

            train_job_name = name_from_base(
                f"{base_prefix}-{HF_MODEL_IDS[2].replace('/', '-').replace('_', '-')}"
            )

            estimator = HuggingFace(
                py_version=PY_VERSION,
                entry_point="train.py",
                transformers_version=TRANSFORMERS_VERSION,
                pytorch_version=PYTORCH_VERSION,
                source_dir=os.path.join("code", "train"),
                hyperparameters=params,
                role=role,
                instance_count=1,
                instance_type=train_instance_type,
                output_path=output_path,
                base_job_name=train_job_name,
                sagemaker_session=sm_session,
            )

            _ = estimator.fit({"training": train_ds_path}, logs=True)

        logger.info("The hyperparameters are as below.")
        logger.info(pformat(params))

    # Combining and deploying models to an endpoint

    infer_image_uri = image_uris.retrieve(
        framework=None,
        region=region_name,
        instance_type=infer_instance_type,
        image_scope=INFER_SCOPE,
        model_id=HF_MODEL_IDS[1],
        model_version=TRAIN_MODEL_VERSION,
    )

    infer_source_uri = script_uris.retrieve(
        model_id=HF_MODEL_IDS[1],
        model_version=TRAIN_MODEL_VERSION,
        script_scope=INFER_SCOPE,
    )

    if model_data is None:
        if use_jumpstart:
            sd_model = estimator.create_model(
                role=role,
                image_uri=infer_image_uri,
                source_dir=infer_source_uri,
                entry_point="inference.py",
            )

        else:
            sd_model = estimator.create_model(
                role=role,
                entry_point="inference.py",
                source_dir=os.path.join("code", "inference", "stable_diffusion"),
            )

    else:
        if use_jumpstart:
            sd_model = Model(
                model_data=model_data,
                role=role,
                image_uri=infer_image_uri,
                source_dir=infer_source_uri,
                entry_point="inference.py",
                sagemaker_session=sm_session,
            )

        else:
            sd_model = HuggingFaceModel(
                model_data=model_data,
                role=role,
                entry_point="inference.py",
                transformers_version=TRANSFORMERS_VERSION,
                pytorch_version=PYTORCH_VERSION,
                py_version=PY_VERSION,
                source_dir=os.path.join("code", "inference", "stable_diffusion"),
                sagemaker_session=sm_session,
            )

    pipeline_model = PipelineModel(
        models=[gpt_model, sd_model],
        role=role,
        sagemaker_session=sm_session,
    )

    _ = pipeline_model.deploy(
        initial_instance_count=1,
        instance_type=infer_instance_type,
        endpoint_name=endpoint_name,
    )

    logger.info(
        "Training the stable diffusion model and deploying an endpoint to %s is complete.",
        endpoint_name,
    )
