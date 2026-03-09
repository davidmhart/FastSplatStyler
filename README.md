# FastSplatStyler
Official Implementation of "Optimization-Free Style Transfer of 3D Gaussian Splats" from WACV 2026.

[Paper](https://openaccess.thecvf.com/content/WACV2026/html/Du_Sablon_Optimization-Free_Style_Transfer_for_3D_Gaussian_Splats_WACV_2026_paper.html) 

[Video](https://www.youtube.com/watch?v=JfOiDYW02EY)

![](example.jpg)

## Example Outputs

Example Outputs can be visualized using the [Antimatter WebGL viewer](https://antimatter15.com/splat/) at the following links.

- Broche: [Original](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/broche-rose-gold_original.splat) and [Stylized](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/broche-rose-gold_style3.splat)
- Crystal Lamp: [Original](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/crystal-lamp-original.splat) and [Stylized](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/crystal-lamp-style2.splat)
- Family Statue: [Original](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/family.ply) and [Stylized](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/family-style-6.splat)
- M60 Tanks: [Original](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/m60.ply) and [Stylized](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/m60-style-31.splat)
- Table: [Original](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/Table.ply) and [Stylized](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/Table_style5.splat)
- Train: [Original](https://antimatter15.com/splat/) and [Stylized](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/train_style1.splat)
- Truck: [Original](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/truck.splat) and [Stylized](https://antimatter15.com/splat/?url=https://huggingface.co/datasets/incrl/fast-splat-styler/resolve/main/truck-style-21.splat)

Example inputs and outputs can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10YmtcCOKGosXfPEi84ho1AfYRYioYo12?usp=drive_link) or [Hugging Face](https://huggingface.co/datasets/incrl/fast-splat-styler/tree/main)

## Install and Run

This work relies heavily on the [Pytorch](https://pytorch.org/) and [Pytorch Geometric](https://www.pyg.org/) libraries.

This code was tested with Python 3.12, Pytorch 2.9.1 (CUDA Toolkit 12.8), and Pytorch Geometric 2.8 on Windows. It was also tested on Mac with the same setup (without Cuda). The provided requirements.txt file comes from the Mac configuration.

This repository relies on a graph networks library that was presented in a [previous work](https://github.com/davidmhart/interpolated-selectionconv/tree/main). The library can be downloaded, with included model weights, at this [google drive link](https://drive.google.com/drive/folders/10YmtcCOKGosXfPEi84ho1AfYRYioYo12?usp=drive_link). Place the "graph_networks" folder in the main directory.

To stylize an example splat, run `python styletransfer_splat.py example-broche-rose-gold.splat --stylePath style_ims/style0.jpg --samplingRate 1.5`. You can change the input splat and style image to your specific use case.

Supports `.splat` and `.ply` files.

## Demo

A gradio demo is also provided. Run `python gradio_app.py` to run it. A HuggingFace Space demo is coming soon.

