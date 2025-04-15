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
