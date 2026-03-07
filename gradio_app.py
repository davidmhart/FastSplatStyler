import gradio as gr
import torch
import os
import tempfile
import shutil
from pathlib import Path
from time import time

# ── Core style-transfer logic (adapted from styletransfer_splat.py) ──────────
import pointCloudToMesh as ply2M
import utils
import graph_io as gio
from clusters import *
import splat_mesh_helpers as splt
import clusters as cl
from torch_geometric.data import Data
from scipy.interpolate import NearestNDInterpolator

from graph_networks.LinearStyleTransfer_vgg import encoder, decoder
from graph_networks.LinearStyleTransfer_matrix import TransformLayer
from graph_networks.LinearStyleTransfer.libs.Matrix import MulLayer
from graph_networks.LinearStyleTransfer.libs.models import encoder4, decoder4


# ── Example assets (place your own files in ./examples/) ─────────────────────
EXAMPLE_SPLATS = [
    ["example-broche-rose-gold.splat",  "style_ims/style2.jpg"],
    ["example-broche-rose-gold.splat",  "style_ims/style6.jpg"],
]


# ── Style-transfer function called by Gradio ─────────────────────────────────
def run_style_transfer(
    splat_file,
    style_image,
    threshold: float,
    sampling_rate: float,
    device_choice: str,
    progress=gr.Progress(track_tqdm=True),
):
    if splat_file is None:
        raise gr.Error("Please upload a 3D Gaussian Splat file (.ply or .splat).")
    if style_image is None:
        raise gr.Error("Please upload a style image.")

    device = device_choice if device_choice == "cpu" else f"cuda:{device_choice}"

    # ── Parameters ────────────────────────────────────────────────────────────
    n = 25
    ratio = 0.25
    depth = 3
    style_shape = (512, 512)

    logs = []

    def log(msg):
        logs.append(msg)
        print(msg)
        return "\n".join(logs)

    # ── 1. Load splat ─────────────────────────────────────────────────────────
    progress(0.05, desc="Loading splat…")
    splat_path = splat_file.name if hasattr(splat_file, "name") else splat_file
    log(f"Loading splat: {splat_path}")

    pos3D_Original, _, colors_Original, opacity_Original, scales_Original, rots_Original, fileType = \
        splt.splat_unpacker_with_threshold(n, splat_path, threshold)

    # ── 2. Gaussian super-sampling ────────────────────────────────────────────
    progress(0.15, desc="Super-sampling…")
    t0 = time()
    if sampling_rate > 1:
        GaussianSamples = int(pos3D_Original.shape[0] * sampling_rate)
        pos3D, colors = splt.splat_GaussianSuperSampler(
            pos3D_Original.clone(), colors_Original.clone(),
            opacity_Original.clone(), scales_Original.clone(), rots_Original.clone(),
            GaussianSamples,
        )
    else:
        pos3D, colors = pos3D_Original, colors_Original
    log(f"Nodes in graph: {pos3D.shape[0]}  ({time()-t0:.1f}s)")

    # ── 3. Graph construction ─────────────────────────────────────────────────
    progress(0.30, desc="Building surface graph…")
    t0 = time()
    style_ref = utils.loadImage(style_image, shape=style_shape)

    normalsNP = ply2M.Estimate_Normals(pos3D, threshold)
    normals = torch.from_numpy(normalsNP)

    up_vector = torch.tensor([[1, 1, 1]], dtype=torch.float)
    up_vector = up_vector / torch.linalg.norm(up_vector, dim=1)

    pos3D = pos3D.to(device)
    colors = colors.to(device)
    normals = normals.to(device)
    up_vector = up_vector.to(device)

    edge_index, directions = gh.surface2Edges(pos3D, normals, up_vector, k_neighbors=16)
    edge_index, selections, interps = gh.edges2Selections(edge_index, directions, interpolated=True)

    clusters, edge_indexes, selections_list, interps_list = cl.makeSurfaceClusters(
        pos3D, normals, edge_index, selections, interps,
        ratio=ratio, up_vector=up_vector, depth=depth, device=device,
    )
    log(f"Graph built  ({time()-t0:.1f}s)")

    # ── 4. Load networks ──────────────────────────────────────────────────────
    progress(0.50, desc="Loading networks…")
    t0 = time()

    enc_ref = encoder4()
    dec_ref = decoder4()
    matrix_ref = MulLayer("r41")

    enc_ref.load_state_dict(torch.load("graph_networks/LinearStyleTransfer/models/vgg_r41.pth",    map_location=device))
    dec_ref.load_state_dict(torch.load("graph_networks/LinearStyleTransfer/models/dec_r41.pth",    map_location=device))
    matrix_ref.load_state_dict(torch.load("graph_networks/LinearStyleTransfer/models/r41.pth",     map_location=device))

    enc = encoder(padding_mode="replicate")
    dec = decoder(padding_mode="replicate")
    matrix = TransformLayer()

    with torch.no_grad():
        enc.copy_weights(enc_ref)
        dec.copy_weights(dec_ref)
        matrix.copy_weights(matrix_ref)

    content = Data(
        x=colors, clusters=clusters,
        edge_indexes=edge_indexes,
        selections_list=selections_list,
        interps_list=interps_list,
    ).to(device)

    style, _ = gio.image2Graph(style_ref, depth=3, device=device)

    enc = enc.to(device)
    dec = dec.to(device)
    matrix = matrix.to(device)
    log(f"Networks loaded  ({time()-t0:.1f}s)")

    # ── 5. Style transfer ─────────────────────────────────────────────────────
    progress(0.70, desc="Running style transfer…")
    t0 = time()

    with torch.no_grad():
        cF = enc(content)
        sF = enc(style)
        feature, _ = matrix(
            cF["r41"], sF["r41"],
            content.edge_indexes[3], content.selections_list[3],
            style.edge_indexes[3],  style.selections_list[3],
            content.interps_list[3] if hasattr(content, "interps_list") else None,
        )
        result = dec(feature, content).clamp(0, 1)

    colors[:, 0:3] = result
    log(f"Stylization done  ({time()-t0:.1f}s)")

    # ── 6. Interpolate back to original resolution ────────────────────────────
    progress(0.88, desc="Interpolating back to original splat…")
    t0 = time()

    interp2 = NearestNDInterpolator(pos3D.cpu(), colors.cpu())
    results_OriginalNP = interp2(pos3D_Original)
    results_Original = torch.from_numpy(results_OriginalNP).to(torch.float32)
    colors_and_opacity_Original = torch.cat(
        (results_Original, opacity_Original.unsqueeze(1)), dim=1
    )
    log(f"Interpolation done  ({time()-t0:.1f}s)")

    # ── 7. Save output ────────────────────────────────────────────────────────
    progress(0.95, desc="Saving output splat…")
    suffix = ".splat" if fileType == "splat" else ".ply"
    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, f"stylized{suffix}")

    splt.splat_save(
        pos3D_Original.numpy(),
        scales_Original.numpy(),
        rots_Original.numpy(),
        colors_and_opacity_Original.numpy(),
        out_path,
        fileType,
    )
    log(f"Saved to: {out_path}")
    progress(1.0, desc="Done!")

    return out_path, "\n".join(logs)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui():
    available_devices = (
        [str(i) for i in range(torch.cuda.device_count())] + ["cpu"]
        if torch.cuda.is_available()
        else ["cpu"]
    )

    with gr.Blocks(
        title="3DGS Style Transfer",
        theme=gr.themes.Soft(primary_hue="violet"),
        css="""
        #title { text-align: center; }
        #subtitle { text-align: center; color: #666; margin-bottom: 1rem; }
        .panel { border-radius: 12px; }
        #run-btn { font-size: 1.1rem; }
        """,
    ) as demo:

        gr.Markdown("# 🎨 3D Gaussian Splat Style Transfer", elem_id="title")
        gr.Markdown(
            "Upload a 3DGS scene and a style image — the app will repaint the splat "
            "with the artistic style of the image and give you a stylized splat to download. "
            "After downloading, you can view your splat with an [online viewer](https://antimatter15.com/splat/).",
            elem_id="subtitle",
        )

        with gr.Row():
            # ── Left column: inputs ───────────────────────────────────────────
            with gr.Column(scale=1, elem_classes="panel"):
                gr.Markdown("### 📂 Inputs")

                splat_input = gr.File(
                    label="3D Gaussian Splat (.ply or .splat)",
                    file_types=[".ply", ".splat"],
                    type="filepath",
                )

                style_input = gr.Image(
                    label="Style Image",
                    type="filepath",
                    height=240,
                )

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    threshold_slider = gr.Slider(
                        minimum=90.0, maximum=100.0, value=99.8, step=0.1,
                        label="Opacity threshold (percentile)",
                        info="Points below this opacity percentile are removed.",
                    )
                    sampling_slider = gr.Slider(
                        minimum=0.5, maximum=3.0, value=1.5, step=0.1,
                        label="Gaussian super-sampling rate",
                        info="Values > 1 add extra samples; 1.0 = no super-sampling.",
                    )
                    device_radio = gr.Radio(
                        choices=available_devices,
                        value=available_devices[0],
                        label="Device",
                    )

                run_btn = gr.Button("🚀 Run Style Transfer", variant="primary", elem_id="run-btn")

            # ── Right column: outputs ─────────────────────────────────────────
            with gr.Column(scale=1, elem_classes="panel"):
                gr.Markdown("### 📥 Output")

                output_file = gr.File(
                    label="Download Stylized Splat",
                    interactive=False,
                )

                log_box = gr.Textbox(
                    label="Progress log",
                    lines=12,
                    max_lines=20,
                    interactive=False,
                    placeholder="Logs will appear here once processing starts…",
                )

        # ── Examples ─────────────────────────────────────────────────────────
        example_splat_paths = [row[0] for row in EXAMPLE_SPLATS]
        example_style_paths = [row[1] for row in EXAMPLE_SPLATS]

        valid_examples = [
            row for row in EXAMPLE_SPLATS
            if os.path.exists(row[0]) and os.path.exists(row[1])
        ]

        if valid_examples:
            gr.Markdown("### 🖼️ Examples")
            gr.Examples(
                examples=valid_examples,
                inputs=[splat_input, style_input],
                label="Click an example to load it",
            )

        # ── Event wiring ──────────────────────────────────────────────────────
        run_btn.click(
            fn=run_style_transfer,
            inputs=[splat_input, style_input, threshold_slider, sampling_slider, device_radio],
            outputs=[output_file, log_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False)
