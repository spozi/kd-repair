# Knowledge Distillation Guidelines for Computer Vision

These recommendations summarize practical findings from the computer-vision knowledge-distillation literature retrieved through Consensus.

## Recommended approach

1. **Establish a strong logits-based baseline first.**

   Train the student with both ground-truth cross-entropy and the teacher's soft targets. Tune the temperature and distillation weight instead of assuming one setting will transfer across datasets. A reasonable initial sweep is temperature `T` in `{2, 4, 8}` and KD weight `lambda` in `{0.25, 0.5, 0.75}` [1].

2. **Try Decoupled Knowledge Distillation (DKD) as the first upgrade.**

   DKD separates target-class knowledge from relationships among non-target classes. It achieved competitive or better performance than more complicated feature-based approaches across CIFAR-100, ImageNet, and MS-COCO while remaining relatively efficient [2].

3. **Do not automatically choose the largest available teacher.**

   A very large teacher-student capacity gap can make knowledge harder to transfer. If the student is dramatically smaller, introduce an intermediate-sized teacher assistant or use progressive, multi-stage distillation [3].

4. **Prioritize teacher quality and confidence, not only accuracy.**

   Teachers with higher-confidence predictions can produce better students. Retain examples the teacher classifies incorrectly: their output distributions may still communicate useful decision-boundary information [4].

5. **Use strong, consistent data augmentation.**

   Augmentation can improve KD for both image classification and object detection. Apply compatible transformations when obtaining teacher and student predictions, and test joint or multiple augmentation strategies rather than relying on one weak transform [5].

6. **Add feature distillation when logits alone plateau.**

   Match semantically corresponding stages rather than blindly forcing individual layers to be identical. Cross-stage or review-based feature transfer is especially helpful when teacher and student depths differ, with demonstrated gains in classification, detection, and instance segmentation [6].

7. **Use task-specific distillation for dense prediction.**

   Classification-style global-logit KD is usually insufficient for object detection or segmentation. Distill foreground-focused features, spatial attention, and relationships among pixels or regions. Structured KD produced a reported 4.1 mAP improvement for Faster R-CNN on COCO in one study [7].

8. **Account for architectural differences.**

   CNN-to-CNN feature matching is easier than ViT-to-CNN or CNN-to-ViT transfer. For heterogeneous pairs, use projection or alignment modules and architecture-specific mechanisms. For a ViT student, a dedicated distillation token is a well-supported option [8].

9. **Tune one component at a time.**

   Compare the following configurations:

   - Student trained without KD
   - Classical logits KD
   - DKD
   - DKD plus a feature or relational loss
   - The best method with stronger augmentation

   Report accuracy or mAP together with parameter count, FLOPs, latency, and training overhead. A distillation method is not useful if it improves accuracy but violates the deployment budget.

## Suggested default recipe

Start with:

> Well-trained, compatible teacher -> supervised cross-entropy + DKD -> temperature and loss-weight sweep -> strong augmentation -> optional cross-stage feature loss

Add task-specific spatial or relational losses for object detection and segmentation.

## References

1. [Distilling the Knowledge in a Neural Network](https://consensus.app/papers/distilling-the-knowledge-in-a-neural-network-hinton-vinyals/ed905c58ea395b17a6b8bfb20603daec/?utm_source=chatgpt) — Geoffrey E. Hinton, O. Vinyals, and J. Dean; 2015; *arXiv*; 25,840 citations. DOI: `10.48550/arxiv.1503.02531`.

2. [Decoupled Knowledge Distillation](https://consensus.app/papers/decoupled-knowledge-distillation-zhao-cui/17a111d2eec1508687419f4b97b82b28/?utm_source=chatgpt) — Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, and Jiajun Liang; 2022; *CVPR 2022*; 929 citations. DOI: `10.1109/cvpr52688.2022.01165`.

3. [Improved Knowledge Distillation via Teacher Assistant](https://consensus.app/papers/improved-knowledge-distillation-via-teacher-assistant-mirzadeh-farajtabar/c03cd2ff5cfe523f8cc7523a2d3d7dc6/?utm_source=chatgpt) — Seyed Iman Mirzadeh et al.; 2019; *AAAI Conference on Artificial Intelligence*; 1,459 citations. DOI: `10.1609/aaai.v34i04.5963`.

4. [Exploring the Knowledge Transferred by Response-Based Teacher-Student Distillation](https://consensus.app/papers/exploring-the-knowledge-transferred-by-responsebased-song-zhou/5b6d691b20e154b88cf8eb29dc9f1f4d/?utm_source=chatgpt) — Liangchen Song et al.; 2023; *ACM International Conference on Multimedia*; 22 citations. DOI: `10.1145/3581783.3612162`.

5. [Multi-perspective analysis on data augmentation in knowledge distillation](https://consensus.app/papers/multiperspective-analysis-on-data-augmentation-in-li-shao/41820a3f324851c1a71afc8914fdfa77/?utm_source=chatgpt) — Wei Li, Shitong Shao, Ziming Qiu, and Aiguo Song; 2024; *Neurocomputing* 583; 10 citations. DOI: `10.1016/j.neucom.2024.127516`.

6. [Distilling Knowledge via Knowledge Review](https://consensus.app/papers/distilling-knowledge-via-knowledge-review-chen-liu/709bac3c681757f7a5b430085edc510c/?utm_source=chatgpt) — Pengguang Chen, Shu Liu, Hengshuang Zhao, and Jiaya Jia; 2021; *CVPR 2021*; 684 citations. DOI: `10.1109/cvpr46437.2021.00497`.

7. [Structured Knowledge Distillation for Accurate and Efficient Object Detection](https://consensus.app/papers/structured-knowledge-distillation-for-accurate-and-zhang-ma/3896cf8c4f455a81936d8a30b005f593/?utm_source=chatgpt) — Linfeng Zhang and Kaisheng Ma; 2023; *IEEE Transactions on Pattern Analysis and Machine Intelligence* 45; 65 citations. DOI: `10.1109/tpami.2023.3300470`.

8. [Training data-efficient image transformers & distillation through attention](https://consensus.app/papers/training-dataefficient-image-transformers-distillation-touvron-cord/0645d19176635a07a2871a8c7ee9a03c/?utm_source=chatgpt) — Hugo Touvron et al.; 2020; venue not listed in the Consensus record; 9,699 citations. DOI: `10.48550/arxiv.2012.12877`.

Citation counts are those reported by Consensus when the research was retrieved.
