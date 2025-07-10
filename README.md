# SOTA-1
Code and materials for the ML SOTA-I lecture.

<details>
<summary> <H2> Week 1 </H2><BR>
Image Classification
</summary>

* BOARD: [https://zoom-x.de/wb/doc/e0_IXOFrS7S3y4Q-Td-APA](https://zoom-x.de/wb/doc/e0_IXOFrS7S3y4Q-Td-APA)


### SotA Links + Materials
* [arxiv.org Preprints](https://arxiv.org/)
    * [Arxiv tag](https://arxivtag.com/)
    * [DL Monitor](https://deeplearn.org/)   
* [Scholar Inbox](https://www.scholar-inbox.com/)
* [AK on Twitter](https://twitter.com/_akhaliq)
* [Papers with Code](https://paperswithcode.com/sota)
* [Hugging Face](https://huggingface.co/models)
* [Zotero](https://www.zotero.org/)

### Image Classification
#### Benchmarks
* [ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)
* [ImageNet100](https://paperswithcode.com/sota/image-classification-on-imagenet-100)
*  ...
  
#### Baseline Models
* ResNet
   * [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
   * [code](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
* Transformer
   * [paper](https://openreview.net/pdf?id=YicbFdNTTy)
   * [code](https://github.com/lucidrains/vit-pytorch) 

#### SOTA CNN
* [ConvNext v2](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf)
   * [code](https://github.com/facebookresearch/ConvNeXt-V2)
   * [ConvNext v1](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)

#### SOTA Transformer
* [Swin v2](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.pdf)
   * [code](https://github.com/microsoft/Swin-Transformer)
   * [Swin v1](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)

</details>

<details>
<summary> <H2> Week 2 </H2><BR>
Image Classification with Foundation Models
</summary>

### Backbones
* [ConvNext v2](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf)
   * [code](https://github.com/facebookresearch/ConvNeXt-V2) 
* [Clipp V2](https://arxiv.org/pdf/2306.15658.pdf)
   * [code](https://github.com/UCSC-VLAA/CLIPA)
   * [CLIP v1 paper](https://arxiv.org/pdf/2103.00020.pdf)
* [Dino V2](https://arxiv.org/pdf/2304.07193.pdf)
   * [code](https://github.com/facebookresearch/dinov2)
   * [DINO V1 paper](https://arxiv.org/pdf/2104.14294.pdf)

### Self-Supervised 
* [Masked AutoEncoder](https://arxiv.org/pdf/2111.06377.pdf)

### SOTA FM Classification
* [Battle of the Backbones](https://openreview.net/pdf?id=1yOnfDpkVe)
   * [code](https://github.com/hsouri/Battle-of-the-Backbones)
* [ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy](https://arxiv.org/pdf/2311.09215.pdf)
   * [code](https://github.com/kirill-vish/Beyond-INet) 

</details>

<details>
<summary> <H2> Week 3 </H2><BR>
Transformers Revised
</summary>

### Background
* [Attetion is all you need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)
* [Why do LLMs attend the first token](https://arxiv.org/pdf/2504.02732)

### Group 1
* [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
* [Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

### Group 2
* [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
* [Layer Normalization](https://arxiv.org/abs/1607.06450)

</details>

<details>
<summary> <H2> Week 4 </H2><BR>
Segmentation I
</summary>

### Benchmarks
* [MS-COCO](https://paperswithcode.com/sota/instance-segmentation-on-coco)
     * [website](https://cocodataset.org/#home)
* [CityScapes](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes)
     * [website](https://www.cityscapes-dataset.com/dataset-overview/)

### Baseline Model
* [U-Net](https://arxiv.org/pdf/1505.04597v1.pdf)
     * [PyTorch Code](https://github.com/milesial/Pytorch-UNet)
     * [Annotated Code](https://nn.labml.ai/unet/index.html)

### SOTA
* [#1 MS-COCO: EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://openaccess.thecvf.com/content/CVPR2023/papers/Fang_EVA_Exploring_the_Limits_of_Masked_Visual_Representation_Learning_at_CVPR_2023_paper.pdf)
     * [code](https://github.com/baaivision/EVA/tree/master/EVA-01)
* [#3 ScityScapes: InternImage: Exploring Large-Scale Vision Foundation Models with
Deformable Convolutions](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_InternImage_Exploring_Large-Scale_Vision_Foundation_Models_With_Deformable_Convolutions_CVPR_2023_paper.pdf)
     * [code](https://github.com/OpenGVLab/InternImage)


</details>
<details>
<summary> <H2> Week 5 </H2><BR>
Segmentation II
</summary>
   
### SOTA
* [Segment Anything (SAM)](https://arxiv.org/pdf/2304.02643.pdf)
     * [code](https://github.com/facebookresearch/segment-anything)
     * [Demo](https://segment-anything.com/demo)
     * [Colab Tutorial](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-anything-with-sam.ipynb)    
* [Segment Everything Everywhere All at Once (SEEM)](https://openreview.net/pdf?id=UHBrWeFWlL)
     * [code](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)
* [Segment Like Me (Slime)](https://arxiv.org/pdf/2309.03179.pdf)
     * [code](https://github.com/aliasgharkhani/SLiMe)
     * [Colab Demo](https://colab.research.google.com/drive/1fpKx6b2hQGEx1GK269vOw_sKeV9Rpnuj?usp=sharing)

</details>

<details>
<summary> <H2> Week 6 </H2><BR>
Depth Estimation 
</summary>
   
### Overview


### Benchmark Monocular
* [NYU-v2 leaderboard](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2)
* [NYU-v2 website](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

### SOTA Monocular
* [Depth Anything](https://arxiv.org/pdf/2401.10891v2.pdf)
   * [code](https://depth-anything.github.io/) 
* [UniDepth - CVPR '24 + NYU-v2 #1](https://arxiv.org/pdf/2403.18913v1.pdf)
   * [code](https://github.com/lpiccinelli-eth/unidepth)    

</details>
<details>

<summary> <H2> Week 7 </H2><BR>
 Visual Question Answering 
</summary>

### Benchmark
* [VQA v2](https://visualqa.org/)
   * [VQA paper](https://arxiv.org/pdf/1505.00468)
   * [VQU v2 paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Goyal_Making_the_v_CVPR_2017_paper.pdf)
   * [Papers with Coder Leaderboard](https://paperswithcode.com/sota/visual-question-answering-on-vqa-v2-test-dev)
 
### SOTA Paper
* [PALI: A JOINTLY-SCALED MULTILINGUAL LANGUAGE-IMAGE MODEL](https://openreview.net/pdf?id=mWVoBz4W0u)
   * [code](https://github.com/kyegomez/PALI) 
* [Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks ](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Image_as_a_Foreign_Language_BEiT_Pretraining_for_Vision_and_CVPR_2023_paper)
   * [code](https://github.com/microsoft/unilm/tree/master/beit3)
   * [BEiT v1](https://openreview.net/pdf?id=p-BhZSz59o4)
   * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)
* [LLVA: Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485)
   * [code](https://github.com/haotian-liu/LLaVA)
   * [Colab Demo](https://colab.research.google.com/drive/1qsl6cd2c8gGtEW1xV5io7S8NHh-Cp1TV?usp=sharing)


</details>

<details>
<summary> <H2> Week 8 </H2><BR>
Genearitve Models I - SOTA GANs 
</summary>

### Benchmark
* [ImageNet 512x512](https://paperswithcode.com/sota/image-generation-on-imagenet-512x512)
* [Flickr-Faces-HQ (FFHQ)](https://paperswithcode.com/sota/image-generation-on-ffhq-256-x-256)
   * [Website](https://github.com/NVlabs/ffhq-dataset) 
#### FID Score
* [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf)
* [Problems with FID](https://openreview.net/pdf?id=mLG96UpmbYz)  

### GAN overview
* [2024 Overview paper](https://iopscience.iop.org/article/10.1088/2632-2153/ad1f77/pdf)
  
### GAN SOTA
* [StyleGAN v2](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670171.pdf)
   * [code](https://github.com/EvgenyKashin/stylegan2-distillation)
   * [StyleGAN v1](https://arxiv.org/pdf/1812.04948)
* [SAN](https://arxiv.org/pdf/2301.12811v4)
   * [code](https://github.com/sony/san)

</details>

<details>
<summary> <H2> Week 9 </H2><BR>
Genearitve Models II - SOTA Diffusion Models
</summary>

### Benchmark (same as week 8)
* [ImageNet 512x512](https://paperswithcode.com/sota/image-generation-on-imagenet-512x512)
* [Flickr-Faces-HQ (FFHQ)](https://paperswithcode.com/sota/image-generation-on-ffhq-256-x-256)
   * [Website](https://github.com/NVlabs/ffhq-dataset) 

### Baseline
* [Original Paper: Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239v2)
  
### Do we really need physical Diffusion ?
* [Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise](https://openreview.net/pdf?id=slHNW9yRie0)
</details>
<details>
<summary> <H2> Week 10 </H2><BR>
Video Generators
</summary>

### "Baseline": Sora
* [OpenAI Sora Tech report](https://openai.com/index/video-generation-models-as-world-simulators/)
* [non-original Paper](https://arxiv.org/pdf/2402.17177)
* [demo video](https://www.youtube.com/watch?v=HK6y8DAPN_0) 

### Latte
* [Paper: Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/pdf/2401.03048v1.pdf)
* [Code](https://github.com/Vchitect/Latte)
* [Demo](https://huggingface.co/spaces/kadirnar/Open-Sora)

### MORA
* [Paper](https://arxiv.org/pdf/2403.13248)
* [Code](https://github.com/lichao-sun/Mora)

</details>

<details>
<summary> <H2> Week 11 </H2><BR>
3D Generators
</summary>



</details>

## Paper for exam
* [ConvNext v1](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)
* [Swin v1](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)
* [CLIP v1 paper](https://arxiv.org/pdf/2103.00020.pdf)
* [DINO V1 paper](https://arxiv.org/pdf/2104.14294.pdf)
* [Masked AutoEncoder](https://arxiv.org/pdf/2111.06377.pdf)
* [Segment Anything (SAM)](https://arxiv.org/pdf/2304.02643.pdf)
* [LLVA: Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485)
* [StyleGAN v1](https://arxiv.org/pdf/1812.04948)
* [Stable Diffusion V1](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
* [VGGT](https://vgg-t.github.io/)
