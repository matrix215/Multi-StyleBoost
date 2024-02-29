# Multi-StyleBoost

## Official PyTorch implementation of the paper "Multi-StyleBoost"

Recent advancements in text-to-image models, such as Stable Diffusion, have demonstrated their
ability to synthesize visual images through natural language prompts. One approach of personalizing text-
to-image models, exemplified by DreamBooth, fine-tunes the pre-trained model by binding unique text
identifiers with a few images of a specific subject. Although existing fine-tuning methods have demonstrated
competence in rendering images according to the styles of famous painters, it is still challenging to learn
to produce images encapsulating distinct art styles due to abstract and broad visual perceptions of stylistic
attributes such as lines, shapes, textures, and colors. In this paper, we present a new fine-tuning method,
called StyleBoost, that equips pre-trained text-to-image models based on full fine-tuning to produce diverse
images in specified styles from text prompts. By leveraging around 15 to 20 images of StyleRef and Aux images
each, our approach establishes a foundational binding of a unique token identifier with a broad realm of
the target style, where the Aux images are carefully selected to strengthen the binding. This dual-binding
strategy grasps the essential concept of art styles and accelerates learning of diverse and comprehensive
attributes of the target style. In addition, we present ways to improve the quality of style and text alignment
through a method called Multi-StyleBoost, which inherits the strategy used in StyleBoost and learns tokens
in multiple. Experimental evaluation conducted on six distinct styles - realism, SureB, anime, romanticism,
cubism, pixel-art - demonstrates substantial improvements in both the quality of generated images and the
perceptual fidelity metrics, such as FID, KID, and CLIP scores.

![multi-singel](https://github.com/matrix215/Multi-StyleBoost/assets/101815603/5d94f816-15c2-42bf-a35e-0f160a591e6d)

# Result
All our results are based on fine-tuning stable-diffusion-v1-5 model. We show results on various categories of images, including scene, person, and style, and with a varying number of training samples. For more generations and comparisons with concurrent methods, please refer to our paper.

![comparison (1)](https://github.com/matrix215/Multi-StyleBoost/assets/101815603/dba39957-8b98-4279-b3d4-1269d519d98a)

## Requirement
- NVIDIA A5000 GPU 

## Getting Started
```bash
  git clone https://github.com/matrix215/Multi-StyleBoost
  cd Multi-StyleBoost
  conda env create -f environment.yaml
  conda activate dream
```
## Multi-StyleBoost training
```bash
  concepts_list = [
        {
            "instance_prompt":      f"zwx style",
            "class_prompt":         "style",
            
            "instance_data_dir":    f"/home/kkko/backup/capston_design/content/data/{instance_style}",
            "class_data_dir":       f"/home/kkko/backup/capston_design/content/data/{class_img_dic}"
        },
        {
            "instance_prompt2":      f"ukj barn",
            "class_prompt2":         "barn",
            
            "instance_data_dir2":    f"/home/kkko/backup/capston_design/content/data/barn",
            "class_data_dir2":       f"/home/kkko/backup/capston_design/content/data/re_barn"
        }
    ]
  ```
Make two concepts using a unique token. like a zwx and ukj




