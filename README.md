# EndoCV2021 teamCodes 
### For the 3rd International Endoscopy Computer Vision Challenge and Workshop (EndoCV2021)

**About:**
The codes for some teams are shared here for reproducibility purposes as part of EndoCV2021 challenge, and upon consent with the participating teams under EndoCV2021. 

**Evaluation repo:** https://github.com/sharib-vision/EndoCV2021-polyp_det_seg_gen

**Challenge title:** Addressing generalisability in polyp detection and segmentation

**Short description:**
Computer-aided detection, localization, and segmentation methods can help improve colonoscopy procedures. Even though many methods have been built to tackle automatic detection and segmentation of polyps, benchmarking and development of computer vision methods remains an open problem. This is mostly due to the lack of datasets or challenges that incorporate highly heterogeneous dataset appealing participants to test for generalisation abilities of the methods. we aim to build a comprehensive, well-curated, and defined dataset from 6 different centres worldwide and provide 5 datasets types that include: i) multi-centre train-test split from 5 centres ii) polyp size-based split (participants should do this by themselves if of interest), iii) data centre wise split, iv) modality split (only test) and v) one hidden centre test. Participants will be evaluated on all types to address strength and weaknesses of each participants’ method.  Both detection bounding boxes and pixel-wise segmentation of polyps will be provided.

**Challenge webpage:** https://endocv2021.grand-challenge.org/

**Analysis (jointly written) and dataset description papers:**

[1] [Assessing generalisability of deep learning-based polyp detection and segmentation methods through a computer vision challenge](https://arxiv.org/abs/2202.12031)

[2] [PolypGen: A multi-center polyp detection and segmentation dataset for generalisability assessment](https://arxiv.org/abs/2106.04463)

## Citation

If you use derivatives of these works and their analysis then please cite, the corresponding method paper and below papers:

    @misc{ali2022-jointJournal,
    title={Assessing generalisability of deep learning-based polyp detection and segmentation methods through a computer vision challenge},
    author    = {Sharib Ali andNoha M. Ghatwary and Debesh Jha and Ece Isik{-}Polat and Gorkem Polat and Chen Yang and Wuyang Li and Adrian Galdran et al.},
    year={2022},
    eprint={2202.12031},
    archivePrefix={arXiv},
    doi ={https://doi.org/10.48550/arXiv.2202.12031}
    }


    @misc{ali2021-datapaper,
    title={PolypGen: A multi-center polyp detection and segmentation dataset for generalisability assessment},
    author    = {Sharib Ali and Debesh Jha and Noha M. Ghatwary and Stefano Realdon and Renato Cannizzaro and Osama E. Salem and Dominique Lamarque  et al.},
    year={2021},
    eprint={2106.04463},
    archivePrefix={arXiv},
    doi ={https://doi.org/10.48550/arXiv.2106.04463}
    }


## Statement & Disclaimer
This project is for research purpose only and may have included third party educational/research codes. The authors do not take any liability regarding any commercial usage.