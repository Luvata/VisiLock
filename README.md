# VisiLock: Authorizing Instruction-based Image Editing with Dual Score Distillation

Official code for **VisiLock** (CVPR 2026), by Van Thanh Le and Yun Fu (Northeastern University).

[[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Le_VisiLock_Authorizing_Instruction-based_Image_editing_with_Dual_Score_Distillation_CVPR_2026_paper.html)]
[[PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Le_VisiLock_Authorizing_Instruction-based_Image_editing_with_Dual_Score_Distillation_CVPR_2026_paper.pdf)]

## Abstract

While open-sourcing instruction-guided image editing models accelerates research, it
surrenders control over their capabilities to anyone who downloads the weights. Existing
protection methods are reactive: they verify ownership after generation, but the underlying
model remains fully functional for unauthorized users. We introduce **VisiLock**, where
access control is baked into model weights, rendering the model unusable without a visual
trigger in the input. The challenge is training a model that retains editing capability for
authorized input and remains unusable for unauthorized input, without destabilizing
training. Naive multi-task objectives create gradient conflicts that collapse training,
while contrastive approaches like FMLock destroy the denoising manifold. We develop **Dual
Score Distillation**, a dual-teacher framework where a degraded teacher defines locked
behavior and an original teacher guides editing quality, eliminating gradient interference
through separate frozen targets. A key risk is that released models could be unlocked
through post-hoc fine-tuning. To prevent this, we initialize the student model from the
degraded teacher so that it begins in a locked state, and only regains editing ability for
authorized inputs via distillation. This impedes adversarial fine-tuning from recovering
full editing capability. Evaluation on InstructPix2Pix shows authorized edits maintain
baseline quality (CLIP-I: 0.821, DINO: 0.726) while unauthorized attempts degrade
substantially (CLIP-I: 0.481, DINO: 0.072) with 41% and 90% drops in image and semantic
similarity. The lock remains robust to key corruptions, spatial perturbations, and
adversarial unlock fine-tuning.

## Citation

```bibtex
@InProceedings{Le_2026_CVPR,
  author    = {Le, Van Thanh and Fu, Yun},
  title     = {VisiLock: Authorizing Instruction-based Image editing with Dual Score Distillation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {15710-15718}
}
```
