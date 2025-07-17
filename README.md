# ProteinDiff - Graph-Conditioned Latent Diffusion (work in progress)
## Structure Conditioned Protein Sequence Generation
### Summary

ProteinDiff is a graph-conditioned latent diffusion model for protein sequence generation given a target backbone structure. It aims to provide a truly generative model for this problem, by reframing it in such a way that the main diffusion model can learn the score of sequence space given a target structure. Here is a summary of the idea in broad strokes:

The input is the N, $C_\alpha$, and C atoms for inference, but for training need all atoms of the protein. 

1. First step is to compute the virtual $C_\beta$ atom from empirical constants. 
2. Next step is to define a local coordinate frame for each residue: 
    - The y-axis is the vector pointing from $C_\alpha$ to the $C_\beta$
    - The x-axis is the vector that is pointing from N to C projected onto the plane normal to the y-axis vector
    - The z-axis is the cross product of the two.  
    - Use the virtual $C_\beta$ if used for frame computation, but the true position for electrostatic field computation later (no $C_\beta$ for glycine).
3. Using this local coordinate frame, we construct a voxel for each residue, with origin at the $C_\beta$, of size $X \times Y \times Z$ (e.g. 16 x 16 x 16, , giving $16\times16\times16\times = 512$ cells, in this case each cell is $0.75^3$ $\AA^3$, making the whole voxel 1728 $\AA^3$)
4. For each atom we assign a partial charge using the AMBER partial charges from ff19SB. using the electric vector field formula, for each residue, we sum the electric effects of all atoms of the residue relative to each cell in the voxel. this creates a voxel for each residue, where each cell is a 3D vector indicating the direction and magnitude of the local electric field at that point. However, in order to avoid the model overfitting to "empty" regions where the amino acid is not present and magnitude is near zero, we normalize the vectors to unit length. This creates a vector field, indicating the direction of the electric field lines (see below for an example visualization). This also helps in providing a consistent input, since all inputs lie between -1 and 1.
5. Here is where it gets interesting.

    - Variational Auto Encoder (VAE)
        
        - The first step is to train a variational autoencoder that learns how to compress the voxelized electric field of each residue, capturing broad and semantic information, such that the it can be reconstructed by the decoder using the sampled latent representation. the vae is pre-trained (along with classifier, but with stop-grad) on the true electric field unit vectors computed from the nearest neighbors' atoms. In this stage, there is no backbone/graph conditioning, the vae only has access to each residues individual voxels, with no cross-talk between residues, besides the initial field calculation.
        
        - Encoder

            - The plan here is to keep it as simple as possible; planning to downsample the voxel 4X with downsampling convolutions. while increasing feature dim (starting point is three features, the electric field unit vectors at each cell). pass the downsampled voxels through a linear layer to get mean and logvars, then sample a latent (Z). Note that a small KL-Divergence term is added to regularize the latent space. 

        - Decoder

            - Decoder performs transposed convolutions to upsample the latent (symmetric to encoder). the output of the decoder is a 3 component vector for each cell, for each residue, each of which is manually normalized to unit length. the loss is then a cosine similarity loss, summed over residues.

    - Graph-Conditioned Latent Diffusion

        - This module operates on the output of the VAE encoder, i.e. the sampled latent. The diffusion module is conditioned on a graph defined by the backbone configuration. to start, will also keep everything as simple as possible. Includes a structure encoder and a diffusion model, for the loss, both trained jointly using MSE on predicted noise, summed over residues.

        - (BackBone) Structure Encoder

            - The backbone is encoded via a Message Passing Neural Network. It is basically a copy of the proteinMPNN encoder. It produces a nodes (V) and edge (E) tensor via message passing operations between neighbors.

        - Graph-Conditioned Latent Diffusion

            - The main denoising network is inspired by Diffusion Transformers (DiT). There are a few preprocessing convolutions that do not compress latent, since the latent is pretty small (4x4x4x4). The convolutions only upsample the feature channel to transformer hidden dim, with no conditioning. The point of this is to allow each cell in the latent voxel to share information about its state with neighboring voxels. This also serves as crude positional encoding for the transformer, since there are only 64 cells, and the size is always the same. 
            
            - Next step is to update the nodes produced by the structure encoder based on the state of the processed latent. We downsample the latent 2X using convolutions w/ SiLU activation, until get $1\times1\times1\times d_\text{model}$ tensor. project to $2\times d_\text{model}$, chunk into $\gamma$ and $\beta$, do $V = \gamma \times V + \beta$, then do PMPNN style message passing, i.e. $V_i$ += $\text{sum}_j[\text{MLP}[V_i, V_j, E_\text{ij}]]$, where the nodes carry backbone info AND info about the abstracted latent. 
            
            - Once the nodes have been updated, use adaLN with conditioning coming from MLP[t, V] on the $4\times4\times4\times d_\text{model}$ tensor, perform self attention on the flattened voxel cells (nor cross talk between residues other than from the node conditioning). 
            
            - The previous two steps are repeated for every DiT layer. This way, the local voxel latents are conditioned on the global backbone configuration, and the global nodes are conditioned on the state of their neighbor's voxel latents.
            
    - Amino Acid Classification

        - This is the simplest module. simply predicts the amino acid class from the reconstructed voxel potentials. Trained alongside vae, but the output of the decoder is detached before going into the classifier, this way the gradients of the classifier do not affect the VAE gradients. Loss is CEL for this module, summed over residues. The 16x16x16x3 voxels are downsampled until get $1\times1\times1\times d_\text{model}$ tensor, project to $num_\text{aa}$, softmax and sample. 

Here is an example of what the model is denoising. Note that these plots are produced using the data space, not the latent space, and no denoising has been done, it is just to give an intuitive idea:

<p align="center">
  <img src="docs/img/noise.png" alt="Arch" width="800"/>
</p>

<p align="center">
  <img src="docs/img/true.png" alt="Arch" width="800"/>
</p>



