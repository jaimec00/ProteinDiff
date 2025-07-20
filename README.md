# ProteinDiff - Graph-Conditioned Latent Diffusion
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
4. For each atom we assign a partial charge using the AMBER partial charges from ff19SB. using the electric vector field formula, for each residue, we sum the electric effects of all atoms of the residue relative to each cell in the voxel. this creates a voxel for each residue, where each cell is a 3D vector indicating the direction and magnitude of the local electric field at that point. However, this introduces a few problems
    - The voxel is large enough to capture the majority of amino acid rotamers, but the downside is that the majority of the voxel will be empty, making it easy to overfit to these empty regions. To prevent this, we normalize the vector of each cell to unit length, making the voxel field purely directional. This does, however, introduce another problem.
    - If the model is trained to denoise unit vectors, these unit vectors do not lie on a Euclidian manifold, they lie on S2 manifold, which would require specialized noise, such as Brownian motion instead of Gaussian noise, and special denoising techniques. To counteract this, we compute the divergence of this directional field, which is a scalar field and lies on a Euclidean manifold. To achieve this, we use finite differences technique.
    - Here is an example of what the scalar field looks like for a single amino acid, along with what gaussian noise looks like in the voxel. All cells within [-0.5,0.5] have been removed for the purposes of visualization.

<p align="center">
  <img src="docs/img/true.png" alt="Arch" width="800"/>
</p>

<p align="center">
  <img src="docs/img/noise.png" alt="Arch" width="800"/>
</p>

5. Here is where it gets interesting.

    - Variational Auto Encoder (VAE)
        
        - The first step is to train a variational autoencoder that learns how to compress the voxelized divergence field of each residue, capturing broad and semantic information, such that the it can be reconstructed by the decoder using the sampled latent representation. the vae is pre-trained (along with classifier, but with stop-grad) on the true divergence voxels. In this stage, there is no backbone/graph conditioning, the vae only has access to each residues individual voxels, with no cross-talk between residues.
        
        - Encoder

            - The plan here is to keep it as simple as possible; planning to downsample the voxel from 1x16x16x16 to 16x1x1x1 with downsampling convolutions. pass the downsampled voxels through a linear layer to get mean and logvars, then sample a latent (Z). Note that a small KL-Divergence term is added to regularize the latent space. 

        - Decoder

            - Decoder performs transposed convolutions to upsample the latent (symmetric to encoder). the output of the decoder is a scalar value for each cell, for each residue. the loss is then mean squared error, summed over residues.

    - Graph-Conditioned Latent Diffusion

        - This module operates on the output of the VAE encoder, i.e. the sampled latent. The diffusion module is conditioned on a graph defined by the backbone configuration.

        - Graph-Conditioned Latent Diffusion

            - The main denoising network is inspired by Diffusion Transformers (DiT). The 16 dim latent vector for each residue is projected to the transformers hidden dimension. Nodes are these projected latents, and edges are computed as in ProteinMPNN, i.e. 16 RBF functions for each inter-residue backbone atom pair projected to transformer hidden dim. These are used as positional encoding of sorts, and are updated at each layer. the nodes perform attention on their nearest neighbors. Both nodes and edges are conditioned on the timestep via adaLN.
            
    - Amino Acid Classification

        - This is the simplest module. simply predicts the amino acid class from the reconstructed voxel potentials. Trained alongside vae, but the output of the decoder is detached before going into the classifier, this way the gradients of the classifier do not affect the VAE gradients. Loss is CEL for this module, summed over residues. The 1x16x16x16 voxels are downsampled until get $1\times1\times1\times d_\text{model}$ tensor, project to $num_\text{aa}$, softmax and sample. 

This project is a work in progress, but at this stage I have implemented all the models. The VAE works very well, and is able to compress the 16x16x16 voxel to a single 16 dim vector, reconstruct with 0.004 MSE, and achieve an unweighted KL-Divergence of 1.7 bits per dim. The classifier also gets 98% top 1 accuracy when classifying the decoder representation, indicating that the latent codes contain enough information for reconstruction, and the decoded voxels contain enough information for accurate classification. We are in the process of testing various configurations for the Diffusion model, with the top runner being the one described above. However, the main bottleneck seems to be in scaling the diffusion model, as the MSE is lower at higher capacity, but takes a very long time to train, and sometimes runs out of memory. Thus, the next steps will probably be to expand to multi-gpu training. DDP is already implemented, but might need to split the model across multiple GPUs if it gets too big. I think this project has great potential, but it will probably take weeks for the diffusion model to learn. Big project, very exciting.