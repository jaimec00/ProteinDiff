import torch
from utils.model_utils.preprocesser import PreProcesser
from data.constants import aa_2_lbl, lbl_2_aa
import numpy as np 
import plotly.graph_objects as go

def main():
    # load a sample pt file
    sample = "4l3o_A"
    # sample_data = f"/work/hcardenas1/projects/ProteinDiff/data/processed/pdb/{sample[1:3]}/{sample}.pt"
    sample_data = f"/home/jaime/Desktop/ProteinDiff/test_data/pdb_2021aug02_sample/pdb/{sample[1:3]}/{sample}.pt"
    pt = torch.load(sample_data, weights_only=True, map_location="cpu")

    # load atomic coordinates, labels, atom mask, and create kp mask (positions w/ no atoms or labels=-1)
    # unsqueeze to simulate batches
    C = pt["xyz"].unsqueeze(0)
    C = C.masked_fill(C.isnan(), 0.0) # get rid of nans
    L = torch.tensor([aa_2_lbl(aa) for aa in pt["seq"]]).unsqueeze(0)
    atom_mask = pt["mask"].unsqueeze(0)

    # instantiate the preprocessor and run it
    voxel_dims = (16,16,16)
    cell_dim = 0.75 # need to define this properly in preprocessor
    prep = PreProcesser(voxel_dims=voxel_dims, cell_dim=cell_dim)
    C_backbone, div = prep(C, L, atom_mask)

    # choose a residue, extract its voxel vector field, and print its AA
    resi = 82
    voxel_vectors = div[0,resi,:,:,:] # Vx,Vy,Vz
    aa = lbl_2_aa(L[0,resi])
    pdb = sample[:4]
    chain = sample[5]
    plot_voxel(voxel_vectors, voxel_dims, cell_dim, f"3D Vector Field of Residue Electrostatics | PDB: {pdb} | Chain: {chain} | Position: {resi} | Amino Acid: {aa}", "/hpc/home/hcardenas1/ProteinDiff/docs/img/div.png")

    # # also show noise
    # noise = torch.randn_like(voxel_vectors)
    # noise_norm = torch.linalg.vector_norm(noise, dim=0, keepdim=True)
    # noise = noise / noise_norm.masked_fill(noise_norm==0,1)
    # plot_voxel(noise, voxel_dims, cell_dim,  "Gaussian Noise", "/hpc/home/hcardenas1/ProteinDiff/docs/img/noise.png")


def plot_voxel(voxel_vectors, voxel_dims, cell_dim, title, path):

    # get voxel dims
    Vx, Vy, Vz = voxel_dims

    # note that the coordinates are in the local frame of the residue
    x, y, z = np.meshgrid(
        cell_dim*(np.arange(Vx) - Vx/2),
        cell_dim*(np.arange(Vy) - Vy/4),
        cell_dim*(np.arange(Vz) - Vz/2),
        indexing='ij'
    )

    # flatten everything for plotly
    X = x.ravel()
    Y = y.ravel()
    Z = z.ravel()

    values = voxel_vectors.ravel()

    fig = go.Figure(data=go.Scatter3d(
        x=X,
        y=Y,
        z=Z,
        mode='markers',
        marker=dict(
            size=6,  # control visual size of the voxel marker
            color=values,
            colorscale=['red', 'white', 'blue'],
            opacity=0.5,
            colorbar=dict(title='Value')
        )
    ))

    # add title and axes, and orient the plot
    fig.update_layout(
        title=title,
        scene=dict(
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=1, z=0)
            ),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )
    )

    # show
    fig.show()

    # fig.write_image(path)

if __name__=="__main__":
    main()