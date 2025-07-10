# ProteinDiff - Graph-Conditioned Latent Diffusion (work in progress)

## summary

this is my last attempt at beating protein mpnn, for now...

this is the idea:

the input is the N, $C_\alpha$, and C atoms for inference, but for training need all atoms of the protein. 

1. first step is to create the virtual $C_\beta$ atom from empirical constants. 
2. next step is to define a local coordinate frame for each residue, the y-axis is the vector pointing from $C_\alpha$ to the $C_\beta$, the x-axis is the vector that is pointing from N to C projected onto the plane normal to the y-axis vector, and the z-axis is the cross product of the two.  use virtual $C_\beta$ for frame computation, but true position for electrostatic potential later (no $C_\beta$ for glycine).
3. using this local coordinate frame, we construct a voxel for each residue, with origin at the $C_\beta$, of size $X \times Y \times Z \AA^3$ (e.g. 9 x 5 x 9 $\AA^3$, in this case each cell is 1 $\AA^3$, giving $9\times5\times9 = 405$ cells)
4. now we compute the topK nearest neighbors of each residue using $C_\alpha$ coordinates. 
5. for each atom we assign a partial charge using the AMBER-defined partial charges. using the the Debye-Huckel screened potential, for each residue, we sum the electric effects of all atoms of all nearest neighbors relative to each cell in the voxel. this creates a voxel for each residue, where the scalar value of each cell is the electric potential at that point.
6. here is where it gets interesting.

    - Variational Auto Encoder (VAE)
        
        - the first step is to train a variational autoencoder that learns how to compress the voxelized electric field of each residue, capturing broad and semantic information, such that the it can be reconstructed by the decoder using the sampled latent representation. the vae is pre-trained (along with classifier, but with stop-grad) on the true electric fields computed from the nearest neighbors' atoms. In this stage, there is no backbone/graph conditioning, the vae only has access to each residues individual voxels, with no cross-talk between residues, besides the initial potential calculation.
        
        - Encoder

            - The plan here is to keep it as simple as possible; planning to downsample the voxel 4X with downsampling convolutions. while increasing feature dim (starting point is one feature, the scalar electric potential at each cell). pass the downsampled voxels through an MLP to get mean and logvars, then sample a latent (Z). Note that a KL-Divergence term is added to regularize the latent space, probably tiny. 

        - Decoder

            - Decoder performs transposed convolutions to upsample the latent. loss is basic MSE, summed over residues.

    - Graph-Conditioned Latent Diffusion

        - this module operates on the output of the VAE encoder, i.e. the sampled latent. The diffusion module is conditioned on a graph defined by the backbone configuration. to start, will also keep everything as simple as possible. Includes a structure encoder and a diffusion model, for the loss, both trained jointly using MSE on predicted noise, summed over residues.

        - (BackBone) Structure Encoder

            - The backbone is encoded via a Message Passing Neural Network. For simplicity, I will literally copy the proteinMPNN encoder. It produces a nodes (V) and edge (E) tensor.

        - Graph-Conditioned Latent Diffusion

            - Will also do my best to keep this as simple as possible. However, not sure if I should use a U-Net, CNN, or maybe (patchified) attention over latent voxel cells, for the main denoising of the latent voxel. the timestep conditioning will probably be implemented using adaLN throughout the module. the graph conditioning is a little trickier, shown below. runs for all the timesteps, probably will use the original $\beta$-scheduler for simplicity. 
            
                - Communication between Latent, Nodes, and Edges
                    
                    - planning to do a unet architecture, which first downsamples the latent all the way tp the bototleneck. before each upsampling operation, use the condensed representation to update the nodes via FiLM, do PMPNN style message passing, i.e. Vi += sum_j[MLP[cat[Vi, Vj, Eij]]], where the nodes carry backbone info AND info about the abstracted latent. after that communicate state of node to condensed latent via FiLM, upsample and cat with the non-conditioned symmetric voxel. this allows the downsampling operation to capture large semantic features about the latent without any conditioning first, then the nodes carry this info and send to other nodes. also allows the final representation to be a mix of conditioned and non-conditioned information.

    - Amino Acid Classification

        - This is the simplest module. simply predicts the amino acid class from the reconstructed voxel potentials. Trained alongside vae, but the output of the decoder is detached before going into the classifier, this way the gradients of the classifier do not affect the VAE gradients. Loss is CEL for this module, summed over residues. 
    

