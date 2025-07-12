# ProteinDiff - Graph-Conditioned Latent Diffusion (work in progress)
## Structure Conditioned Protein Sequence Generation
### summary

the input is the N, $C_\alpha$, and C atoms for inference, but for training need all atoms of the protein. 

1. first step is to create the virtual $C_\beta$ atom from empirical constants. 
2. next step is to define a local coordinate frame for each residue, the y-axis is the vector pointing from $C_\alpha$ to the $C_\beta$, the x-axis is the vector that is pointing from N to C projected onto the plane normal to the y-axis vector, and the z-axis is the cross product of the two.  use virtual $C_\beta$ for frame computation, but true position for electrostatic potential later (no $C_\beta$ for glycine).
3. using this local coordinate frame, we construct a voxel for each residue, with origin at the $C_\beta$, of size $X \times Y \times Z \AA^3$ (e.g. 9 x 5 x 9 $\AA^3$, in this case each cell is 1 $\AA^3$, giving $9\times5\times9 = 405$ cells)
4. now we compute the topK nearest neighbors of each residue using $C_\alpha$ coordinates. 
5. for each atom we assign a partial charge using the AMBER-defined partial charges. using a modified electric field formula, for each residue, we sum the electric effects of all atoms of all nearest neighbors relative to each cell in the voxel. this creates a voxel for each residue, where each cell is a 3D vector indicating the direction and magnitude of the electric field at that point.
6. here is where it gets interesting.

    - Variational Auto Encoder (VAE)
        
        - the first step is to train a variational autoencoder that learns how to compress the voxelized electric field of each residue, capturing broad and semantic information, such that the it can be reconstructed by the decoder using the sampled latent representation. the vae is pre-trained (along with classifier, but with stop-grad) on the true electric fields computed from the nearest neighbors' atoms. In this stage, there is no backbone/graph conditioning, the vae only has access to each residues individual voxels, with no cross-talk between residues, besides the initial potential calculation.
        
        - Encoder

            - The plan here is to keep it as simple as possible; planning to downsample the voxel 4X with downsampling convolutions. while increasing feature dim (starting point is one feature, the scalar electric potential at each cell). pass the downsampled voxels through an MLP to get mean and logvars, then sample a latent (Z). Note that a KL-Divergence term is added to regularize the latent space, probably tiny. 

        - Decoder

            - Decoder performs transposed convolutions to upsample the latent (symmetric to encoder). loss is basic MSE, summed over residues.

    - Graph-Conditioned Latent Diffusion

        - this module operates on the output of the VAE encoder, i.e. the sampled latent. The diffusion module is conditioned on a graph defined by the backbone configuration. to start, will also keep everything as simple as possible. Includes a structure encoder and a diffusion model, for the loss, both trained jointly using MSE on predicted noise, summed over residues.

        - (BackBone) Structure Encoder

            - The backbone is encoded via a Message Passing Neural Network. For simplicity, I will literally copy the proteinMPNN encoder. It produces a nodes (V) and edge (E) tensor.

        - Graph-Conditioned Latent Diffusion

            - I am thinking of copying DiT. I will also have a few preprocessing convolutions, but they will probably not downsample, since the latent is pretty small (probably 4x4x4x4) i think the conv operations will be to upsample the feature channel only to transformer hidden dim, with no conditioning. then do downsampling conv w/ SiLU activation, keeping the feature dim as is, until get 1x1x1xd_model. project to 2*d_model, chunk into gamma and beta, do V = gamma*V + beta, then do PMPNN style message passing, i.e. Vi += sum_j[MLP[Vi, Vj, Eij]], where the nodes carry backbone info AND info about the abstracted latent. once nodes have been updated, use adaLN with conditioning coming from MLP[t, V] on the 4x4x4xdmodel tensor, perform self attention. repeat for every DiT layer.  
            
    - Amino Acid Classification

        - This is the simplest module. simply predicts the amino acid class from the reconstructed voxel potentials. Trained alongside vae, but the output of the decoder is detached before going into the classifier, this way the gradients of the classifier do not affect the VAE gradients. Loss is CEL for this module, summed over residues. Will probably do something similar to DiT abstraction of the latent, ie conv downsample until get 1x1x1xdmodel tensor, project to num_aa, softmax and sample. 

Here is an example of what exactly the model is denoising. Note that these plots are produced using the data space, not the latent space, and no denoising has been done, it is just to give you an idea:

<p align="center">
  <img src="docs/img/gauss_noise.png" alt="Arch" width="800"/>
</p>

<p align="center">
  <img src="docs/img/true_4l3o_A_resi5_R.png" alt="Arch" width="800"/>
</p>



