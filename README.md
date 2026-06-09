# Mitigating scalability challenges in LUT-based neural networks via pruning optimisations

![image](figs/Overview.png)

This is repository shows **training** and **deployment** example of using LUT-MU based CNN models. 
The implementation of LUT-MU is based on [Halutmatmul](https://github.com/joennlae/halutmatmul.git), [Brevitas](https://github.com/Xilinx/brevitas) and [FINN](https://github.com/Xilinx/finn) workflow:
-  **Halutmatmul** enables the training of LUT-MU based NN by using differentiable operator.
 - **Brevitas** is the front-end of FINN, which supports Quantised Awared Training (QAT).
 - **FINN** provides End-to-End deployment (convert **Brevitas** qonnx to bitstream).

## Main ideas of LUT-MU
![image](figs/Challenge.png)
 - The index-based clustering mechanism of Product Quantisation (e.g., MADDNESS) can **restrict dot-product computations in successive layers to selected indices**, offering a natural opportunity for computational pruning in neural networks. 
 - The codebook-independent memory access partten allows LUT-MU **colum-wisely partion LUT into distributed ROMs**, which alleviates the II bottleneck derived from incoherence memory access.


## Quick start

- Follow the [manual](./train) to train LUT-MU based CNN.

- Follow the [manual](./deploy) to generate and deploy bitstream on FPGA. (🚧 **TODO** 🚧)