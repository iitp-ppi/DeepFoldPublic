# Loss Functions

The network is trained end-to-end, with gradients coming from the main Frame Alionged Point Error (FAPE) loss $\mathcal{L}_\mathrm{FAPE}$ and a number of auxiliary losses.

## Frame alionged point error (FAPE)

The Frame Alinged Point Error (FAPE) scores a set of predicted atom coordinates $\{ \vec{x}_j \}$ under a set of predicted local frames $\{ T_i \}$ against the corresponding ground truth atom coordinates $\{ \vec{x}^0_j \}$ and ground truth local frames $\{T^0_i\}$.
The final FAPE loss scores all atoms in all backbone nad side chain frames.
