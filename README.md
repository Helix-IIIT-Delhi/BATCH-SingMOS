<div align="center">

# BATCH (Bhattacharya Distance)
## Investigating the Reasonable Effectiveness of Speaker Pre-Trained Models and their Synergistic Power for SingMOS Prediction
[![arXiv](https://img.shields.io/badge/arXiv-2506.02232-b31b1b.svg)](https://arxiv.org/pdf/2506.02232), [INTERSPEECH 2025]

</div>

---

**Abstract**  
In this study, we focus on Singing Voice Mean Opinion Score
 arXiv:2506.02232v1  [eess.AS]  2 Jun 2025
 (SingMOS) prediction. Previous research have shown the perfor
mance benefit with the use of state-of-the-art (SOTA) pre-trained
 models (PTMs). However, they haven’t explored speaker recog
nition speech PTMs (SPTMs) such as x-vector, ECAPA and
 we hypothesize that it will be the most effective for SingMOS
 prediction. We believe that due to their speaker recognition pre
training, it equips them to capture fine-grained vocal features
 (e.g., pitch, tone, intensity) from synthesized singing voices in a
 much more better way than other PTMs. Our experiments with
 SOTA PTMsincluding SPTMs and music PTMs validates the
 hypothesis. Additionally, we introduce a novel fusion frame
work, BATCH that uses Bhattacharya Distance for fusion of PTMs.
 Through BATCH with the fusion of speaker recognition SPTMs,
 we report the topmost performance comparison to all the indi
vidual PTMs and baseline fusion techniques as well as setting
 SOTA.

---
## Source code to be released soon—stay tuned.
---

<div align="center">
    <img src="./sing_fake_archi1.jpg" alt="Pipeline" style="width: 50%;"/>
</div>

---
## Citation
If you find this work useful, please consider citing us:

```bibtex

@misc{phukan2025sourceattributionsingingvoice,
    title={Towards Source Attribution of Singing Voice Deepfake with Multimodal Foundation Models}, 
    author={Orchid Chetia Phukan and Girish and Mohd Mujtaba Akhtar and Swarup Ranjan Behera and Priyabrata Mallick and Pailla Balakrishna Reddy and Arun Balaji Buduru and Rajesh Sharma},
    year={2025},
    eprint={2506.03364},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={https://arxiv.org/abs/2506.03364}, 
}