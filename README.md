# Lensa-for-All
- - -
![avatars-01](assets/avatars-01.png)
This is **Lensa for All**, an application that creates personalized images by fine-tuning the image generation model with just a few photos. If you have 4-20 photos and an AWS account ID, you can create as many [*Lensa*](https://prisma-ai.com/lensa)-style AI avatars as you want for a fraction of the price.
* The application enables the generation of images of you, your family and friends, your pets, etc. by fine-tuning the [*Stable Diffusion*](https://stability.ai/blog/stable-diffusion-public-release) using the *Dreambooth* proposed by Google in their paper ['DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation (2022)'](https://dreambooth.github.io/).
* This used [*Amazon SageMaker*](https://aws.amazon.com/sagemaker/) Services and 🤗 [*Diffusers*](https://huggingface.co/docs/diffusers/index) and [*Accelerate*](https://huggingface.co/docs/accelerate/index) to optimize infrastructure requirements. The model training takes about an hour, which is the equivalent of $1.212 as of April 2023 for `ml.g5.2xlarge` in the Virginia region of the US. (It's less if you use `ml.g4dn.2xlarge`.)
* This also supports auto-prompting using [Gustavosta's *MagicPrompt*](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion) based on *GPT2*. 

## Prerequisites
- - -
### Requirements
* To use AWS services, you must first have an AWS account and have your credentials set up through the AWS CLI. For more information, see [the following documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).
* To use 🤗 *Hub*, you must have a user access token, see [the following documentation](https://huggingface.co/docs/hub/security-tokens) for more information.
* It's nice to have [*WandB*](https://wandb.ai/site)'s API key so you can monitor your model's training progress, but it's not mandatory.

### Installation
1. Git clone this repository in a directory of your choice.  
`git clone https://github.com/youngmki/lensa-for-all.git`  
2. Install the dependency libraries. I recommend using a virtual environment like [*Conda*](https://docs.conda.io/en/latest/).  
`pip install -r requirements.txt`  

### Configuration
* Setup a configuration file named `config/config.yaml`. Entering `hf_token`, `subject_name` and `class_name` are required.
* Enter `iam_profile_name`, `iam_role`, `s3_bucket` and `wandb_api_key` if you need to specify them yourself.
* For `with_prior_preservation`, `train_text_encoder` and `max_steps`, read the advice for fine-tuning below and enter the appropriate values. 
* Enter `train_instance_type` as `ml.g5.2xlarge` if you are using `with_prior_preservation` or `train_text_encoder`, or `ml.g4dn.2xlarge` otherwise.
```yaml
environment:
  iam_profile_name: default
  region_name: us-east-1
  iam_role:  # Specify an IAM role that runs the SageMaker jobs. If left blank, it will be created and run.
  ebs_dataset_dir: input_directory
  s3_bucket:  # Specify a S3 bucket to store data and artifacts in. If left blank, it will be created and run.
  s3_base_prefix: lensa-for-all
  s3_dataset_prefix: input_directory
  hf_token:  # The Hugging Face token input is mandatory.
  wandb_api_key:  # The WandB API key input is optional.

input:
  subject_name: sks  # Enter the name of the subject you want to train the model on. It's okay to be arbitrary (e.g., sks).
  class_name: person  # Enter the name of the class to which the subjects you want to train the model belong. 

model:
  use_jumpstart: False  # Select True to use Amazon SageMaker Jumpstart, or False to use the Hugging Face libraries directly.
  model_data:  # If you already have a model artifact in S3 that you've trained on, enter the prefix here. Otherwise, leave it blank.
  with_prior_preservation: True
  train_text_encoder: True  # Not applicable if you are using Amazon SageMaker Jumpstart.
  max_steps: 300
  batch_size: 1
  learning_rate: 1e-06
  tune_params: False  # If you are using the Amazon SageMaker Tuner to tune hyperparameters, leave it True or False.
  max_tuning_jobs: 7
  train_instance_type: ml.g5.2xlarge
  infer_instance_count: 1
  infer_instance_type: ml.g4dn.2xlarge
  sm_endpoint_name: lensa-for-all
```

## Usage
- - -
1. To train a model of your desired subject, please save 4-20 photos in your project directory as shown below. If possible, I recommend using a jpg/jpeg/png format of 512 × 512 pixels or larger. Also, modify the `config/config.yaml` file to what you want.
  
```bash
input_directory  
├── instance_image_01.jpg  
├── instance_image_02.jpg
├── instance_image_03.jpg
└── ...
```
2. In the terminal, run the following script to upload the saved photos to the S3 bucket.

```bash
python upload_files.py
```

3. In the terminal, run the following script to fine-tune the model and deploy it to a *SageMaker Endpoint*.  This should take just over an hour.

```bash
python train_and_deploy_models.py
```

4. Run the *Jupyter Notebook* file as follows and follow the instructions there. When you're done using the model, be sure to remove the *SageMaker Endpoint* to avoid unnecessary costs.

```bash
jupyter notebook generate_images.ipynb
```

![avatars-02](assets/avatars-02.png)
## Tips for Successful Fine-tuning
- - -
* Start with a good number of high-quality training images (4 to 20). If you are training on human faces, you may need more images. I recommend [12 faces, 5 upper body shots, and 3 full body shots](https://techpp.com/2022/10/10/how-to-train-stable-diffusion-ai-dreambooth/) taken in [a variety of backgrounds, lighting, gazes, and facial expressions](https://github.com/JoePenna/Dreambooth-Stable-Diffusion).
* *Dreambooth* tends to overfit quickly, so I recommend setting the learning rate low (1e-06 to 2e-06) and gradually increasing the number of training steps until you are satisfied with the results. As a guide, if the generated images are noisy or degraded and look almost identical to the photos in your training set, they're overfitted. (If you've set up *WandB*, look at the validation images that are generated and make a judgment call. For example, if the Eiffel Tower that you set as the background in the prompt starts to disappear, that's evidence of overfitting). To address this, reduce the number of training steps. 
* When learning non-human objects, such as dogs and cats, you typically need 200 to 400 steps of training. However, if you're learning a human face, you may need more training, perhaps 800 to 1200 steps.
* If you're not training human faces, you don't need to use *prior preservation loss*, and it won't have a significant impact on performance. However, you should use it for human faces training.
Also, fine-tuning the text encoder along with the image generator has been shown to produce the best image quality. However, using priors and training the text encoder requires more GPU memory (at least `ml.g5.2xlarge`). If you use techniques like [*8-bit Adam*](https://arxiv.org/abs/2110.02861), half precision training, or [gradient accumulation](https://arxiv.org/abs/1710.02368), you can get by with `ml.g4dn.2xlarge`.

## Roadmap
- - -
- [ ] Provides quality prompt templates from [*PromptHero*](https://prompthero.com/) and more.
- [ ] Enable early stopping of model training based on [*FID*](https://arxiv.org/pdf/1706.08500.pdf) scores.
- [ ] Instead of manually generating images using *Jupyter Notebook*, create a *Python* script that automatically generates the desired number of images.
- [ ] Redesign each workflow to be object-oriented, and organize and register them as a *Python* package.

## References
- - -
* [‘모두를 위한 렌사’ 만들기 (Korean)](https://medium.com/@aldente0630/%EB%AA%A8%EB%91%90%EB%A5%BC-%EC%9C%84%ED%95%9C-%EB%A0%8C%EC%82%AC-%EB%A7%8C%EB%93%A4%EA%B8%B0-e445adbe445d), [Make “Lensa for All” (English)](https://medium.com/@aldente0630/creating-a-lensa-for-all-english-29872bf4d846)
* [*Stable Diffusion* on *Amazon SageMaker*](https://www.philschmid.de/sagemaker-stable-diffusion)
* [Fine-Tune Text-to-Image *Stable Diffusion* Models with *Amazon SageMaker JumpStart*](https://aws.amazon.com/blogs/machine-learning/fine-tune-text-to-image-stable-diffusion-models-with-amazon-sagemaker-jumpstart/)
* [Training *Stable Diffusion* with *Dreambooth* Using 🧨 *Diffusers*](https://huggingface.co/blog/dreambooth)
* [How to Fine-tune *Stable Diffusion* Using *Dreambooth*](https://towardsdatascience.com/how-to-fine-tune-stable-diffusion-using-dreambooth-dfa6694524ae)
