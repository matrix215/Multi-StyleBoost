# Multi-StyleBoost

## This is a PyTorch implementation of the paper "Multi-StyleBoost"

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

# Requirement
- Use requirements.txt
- I used NVIDIA A5000 GPU x4

# How to Start
-First 
```bash
  python training_multi.py
```

