import pointCloudToMesh as ply2M
import argparse
import utils
import graph_io as gio
from clusters import *
#from tqdm import tqdm,trange
import splat_mesh_helpers as splt
import clusters as cl
from torch_geometric.data import Data
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import pointCloudToMesh as plyToMesh
import plotly.graph_objects as go
import pyvista as pv

from time import time

from graph_networks.LinearStyleTransfer_vgg import encoder,decoder
from graph_networks.LinearStyleTransfer_matrix import TransformLayer

from graph_networks.LinearStyleTransfer.libs.Matrix import MulLayer
from graph_networks.LinearStyleTransfer.libs.models import encoder4, decoder4

#import matplotlib.pyplot as plt


def styletransfer_with_filtering_sampling(filename, stylePath, outPath, device = 'cpu', threshold=99.9, samplingRate=1.5, displayPointCloud = False):
    
    print("Running on device:", device)

    n = 25
    style_ref = utils.loadImage(stylePath, shape=(256*2,256*2))
    ratio=.25
    depth = 3

    pos3D_Original, _, colors_Original, opacity_Original, scales_Original, rots_Original, fileType = splt.splat_unpacker_with_threshold(n, filename, threshold)
    
    time1_start = time()

    #plyToMesh.graph_Points(pos3D_Original, torch.clamp(colors_Original, 0, 1))
    if samplingRate > 1:
        GaussianSamples = int(pos3D_Original.shape[0]*samplingRate)
        pos3D, colors = splt.splat_GaussianSuperSampler(pos3D_Original.clone(), colors_Original.clone(), opacity_Original.clone(), scales_Original.clone(), rots_Original.clone(), GaussianSamples)
    else:
        pos3D, colors = pos3D_Original, colors_Original
    #plyToMesh.graph_Points(pos3D, torch.clamp(colors, 0, 1))
    #plyToMesh.graph_Points(pos3D_Original, torch.clamp(colors_Original, 0, 1))
    
    time1_end = time()
    
    print("Number of nodes in the graph:", pos3D.shape[0])
    
    print(f"Time taken for Gaussian Super Sampling: {time1_end - time1_start}")


    if (displayPointCloud):
        #point cloud
        point_cloud = pv.PolyData(pos3D.numpy())

        # Add colors to the point data
        point_cloud.point_data['colors'] = torch.clamp(colors, 0, 3).numpy()

        # Plot the point cloud
        plotter = pv.Plotter()
        plotter.add_points(point_cloud, scalars='colors', rgb=True, point_size=0.05)
        plotter.show_axes()
        plotter.show()

    time2_start = time()

    #find normals
    normalsNP = ply2M.Estimate_Normals(pos3D, threshold)
    normals = torch.from_numpy(normalsNP)

    #print("Time to compute normals:", time() - time2_start)
    
    up_vector = torch.tensor([[1,1,1]],dtype=torch.float)
    #up_vector = 2*torch.rand((1,3))-1
    up_vector = up_vector/torch.linalg.norm(up_vector,dim=1)

    pos3D.to(device)
    colors.to(device)
    normals.to(device)
    up_vector.to(device)

    # Build initial graph
    #edge_index are neighbors of a point, directions are the directions from that point
    edge_index,directions = gh.surface2Edges(pos3D,normals,up_vector,k_neighbors=16)
    #directions need to be turned into selections "W sub n" from the star-like coordinate system from Dr. Hart's github interpolated-selectionconv
    edge_index,selections,interps = gh.edges2Selections(edge_index,directions,interpolated=True)

    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list, interps_list = cl.makeSurfaceClusters(pos3D,normals,edge_index,selections,interps,ratio=ratio,up_vector=up_vector,depth=depth,device=device)
    #clusters, edge_indexes, selections_list, interps_list = cl.makeMeshClusters(pos3D,mesh,edge_index,selections,interps,ratio=ratio,up_vector=up_vector,depth=depth,device=device)

    time2_end = time()
    print(f"Time taken for graph construction: {time2_end - time2_start}")

    time3_start = time()

    # Make final graph and metadata needed for mapping the result after going through the network
    content = Data(x=colors,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=interps_list)
    content_meta = Data(pos3D=pos3D)

    style,_ = gio.image2Graph(style_ref,depth=3,device=device)

    # Load original network
    enc_ref = encoder4()
    dec_ref = decoder4()
    matrix_ref = MulLayer('r41')

    enc_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/vgg_r41.pth'))
    dec_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/dec_r41.pth'))
    matrix_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/r41.pth',map_location=torch.device(device)))

    # Copy weights to graph network
    enc = encoder(padding_mode="replicate")
    dec = decoder(padding_mode="replicate")
    matrix = TransformLayer()

    with torch.no_grad():
        enc.copy_weights(enc_ref)
        dec.copy_weights(dec_ref)
        matrix.copy_weights(matrix_ref)

    content = content.to(device)
    style = style.to(device)
    enc = enc.to(device)
    dec = dec.to(device)
    matrix = matrix.to(device)

    # Run graph network
    with torch.no_grad():
        cF = enc(content)
        sF = enc(style)
        feature,transmatrix = matrix(cF['r41'],sF['r41'],
                                        content.edge_indexes[3],content.selections_list[3],
                                        style.edge_indexes[3],style.selections_list[3],
                                        content.interps_list[3] if hasattr(content,'interps_list') else None)
        result = dec(feature,content)
        result = result.clamp(0,1)

    colors[:, 0:3] = result

    time3_end = time()
    print(f"Time taken for stylization: {time3_end - time3_start}")

    if (displayPointCloud):
        #point cloud
        point_cloud = pv.PolyData(pos3D.numpy())

        # Add colors to the point data
        point_cloud.point_data['colors'] = torch.clamp(colors, 0, 3).numpy()

        # Plot the point cloud
        plotter = pv.Plotter()
        plotter.add_points(point_cloud, scalars='colors', rgb=True, point_size=0.25)
        plotter.show_axes()
        plotter.show()

    time4_start = time()

    #create the interpolator
    interp2 = NearestNDInterpolator(pos3D.cpu(), colors.cpu())
    results_OriginalNP = interp2(pos3D_Original)
    results_OriginalNP64 = torch.from_numpy(results_OriginalNP)
    results_Original = results_OriginalNP64.to(torch.float32)

    colors_and_opacity_Original = torch.cat((results_Original, opacity_Original.unsqueeze(1)), dim=1)

    time4_end = time()
    print(f"Time taken for interpolation: {time4_end - time4_start}")

    # Save/show result
    splt.splat_save(pos3D_Original.numpy(), scales_Original.numpy(), rots_Original.numpy(), colors_and_opacity_Original.numpy(), outPath, fileType)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=str,
        default=''
    )
    parser.add_argument(
        "--stylePath",
        type=str,
        default="style_ims/style0.jpg"
    )
    parser.add_argument(
        "--outPath",
        type=str,
        default='output.splat'
    )
    parser.add_argument(
        "--device",
        default= 0 if torch.cuda.is_available() else "cpu",
        choices=list(range(torch.cuda.device_count())) + ["cpu"] or ["cpu"]
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=99.8
    )
    parser.add_argument(
        "--samplingRate",
        type=float,
        default=1.5
    )
    parser.add_argument(
        "--displayPointCloud",
        action='store_true'
    )
    args = parser.parse_args()
    styletransfer_with_filtering_sampling(**vars(args))


if __name__ == "__main__":
    main()
