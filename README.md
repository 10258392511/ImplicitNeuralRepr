# Implicit Neural Representation for Accelerated Cardiac MRI Reconstruction

Please first install `PyTorch` following the [official website](https://pytorch.org/). Then please install other 
dependencies by:
```bash
git clone https://github.com/10258392511/ImplicitNeuralRepr.git
cd ImplicitNeuralRepr
pip3 install -e .
```
### Global: 2D + Time + Regularization
<table align="center" id="global-2d-time-reg">
    <tr>
        <th>&lambda;</th>
        <th>Mag<span style="color: white;">.</span></th>
        <th >Phase</th>
        <th>Mag<span style="color: white;">.</span> @ T / 2</th>
        <th>Phase @ T / 2</th>
        <th>Mag<span style="color: white;">.</span> @ H / 2</th>
    </tr>
    <tbody align="center">
        <tr>
            <td>Original</td>
            <td><img src="readme_images/global_2d_time_reg/original/mag.gif" alt="original mag"></td>
            <td><img src="readme_images/global_2d_time_reg/original/phase.gif" alt="original phase"></td>
            <td><img src="readme_images/global_2d_time_reg/original/half_T_mag.png" alt="original mag at half T"></td>
            <td><img src="readme_images/global_2d_time_reg/original/half_T_phase.png" alt="original phase at half T"></td>
            <td><img src="readme_images/global_2d_time_reg/original/half_H_mag.png" alt="original mag at half H"></td>
        </tr>
        <tr>
            <td>ZF</td>
            <td><img src="readme_images/global_2d_time_reg/ZF/mag.gif" alt="ZF mag"></td>
            <td><img src="readme_images/global_2d_time_reg/ZF/phase.gif" alt="ZF phase"></td>
            <td><img src="readme_images/global_2d_time_reg/ZF/half_T_mag.png" alt="ZF mag at half T"></td>
            <td><img src="readme_images/global_2d_time_reg/ZF/half_T_phase.png" alt="ZF phase at half T"></td>
            <td><img src="readme_images/global_2d_time_reg/ZF/half_H_mag.png" alt="ZF mag at half H"></td>
        </tr>
        <tr>
            <td>-5</td>
            <td><img src="readme_images/global_2d_time_reg/lam_-5/mag.gif" alt="lam_-5 mag"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_-5/phase.gif" alt="lam_-5 phase"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_-5/half_T_mag.png" alt="lam_-5 mag at half T"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_-5/half_T_phase.png" alt="lam_-5 phase at half T"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_-5/half_H_mag.png" alt="lam_-5 mag at half H"></td>
        </tr>
        <tr>
            <td>-2.5</td>
            <td><img src="readme_images/global_2d_time_reg/lam_-2_5/mag.gif" alt="lam_-2_5 mag"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_-2_5/phase.gif" alt="lam_-2_5 phase"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_-2_5/half_T_mag.png" alt="lam_-2_5 mag at half T"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_-2_5/half_T_phase.png" alt="lam_-2_5 phase at half T"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_-2_5/half_H_mag.png" alt="lam_-2_5 mag at half H"></td>
        </tr>
        <tr>
            <td>0</td>
            <td><img src="readme_images/global_2d_time_reg/lam_0/mag.gif" alt="lam_0 mag"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_0/phase.gif" alt="lam_0 phase"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_0/half_T_mag.png" alt="lam_0 mag at half T"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_0/half_T_phase.png" alt="lam_0 phase at half T"></td>
            <td><img src="readme_images/global_2d_time_reg/lam_0/half_H_mag.png" alt="original mag at half H"></td>
        </tr>
    </tbody>
</table>

<div align="center">
    <strong>Reconstruction with different &lambda;, taking &lambda; as coordinate.
Larger &lambda; results in smoother signals in temporal dimension. However, these reconstructions lose details and texture.</strong>
</div>

<br>

### LIIF: 2D vs 3D Convolutional Encoder
<table align="center" id="liif-2d-vs-3d-conv-enc">
    <thead>
        <tr>
            <th>Config</th>
            <th>Mag<span style="color: white;">.</span></th>
            <th>Mag<span style="color: white;">.</span> @ T / 2</th>
            <th>Mag<span style="color: white;">.</span> Error @ T / 2</th>
            <th>Mag<span style="color: white;">.</span> @ H / 2</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td>Original</td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/original/mag.gif" alt="original"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/original/half_T_mag.png" alt="original at half T"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/original/half_T_mag_error.png" alt="original, error at half T"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/original/half_H_mag.png" alt="original at half H"></td>
        </tr>
        <tr>
            <td>ZF</td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/ZF/mag.gif" alt="ZF"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/ZF/half_T_mag.png" alt="ZF at half T"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/ZF/half_T_mag_error.png" alt="ZF, error at half T"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/ZF/half_H_mag.png" alt="ZF at half H"></td>
        </tr>
        <tr>
            <td>2D, C = 8, Sine</td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/2d_8/mag.gif" alt="2D, C = 8,"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/2d_8/half_T_mag.png" alt="2D, C = 8, Sine, at half T"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/2d_8/half_T_mag_error.png" alt="2D, C = 8, Sine, error at half T"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/2d_8/half_H_mag.png" alt="2D, C = 8, Sine, at half H"></td>
        </tr>
        <tr>
            <td>2D, C = 64, Sine</td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/2d_64/mag.gif" alt="2D, C = 64, Sine"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/2d_64/half_T_mag.png" alt="2D, C = 64, Sine at half T"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/2d_64/half_T_mag_error.png" alt="2D, C = 64, Sine error at half T"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/2d_64/half_H_mag.png" alt="2D, C = 64, Sine at half H"></td>
        </tr>
        <tr>
            <td>3D, C = 16, Sine</td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/3d_16/mag.gif" alt="3D, C = 16, Sine"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/3d_16/half_T_mag.png" alt="3D, C = 16, Sine at half T"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/3d_16/half_T_mag_error.png" alt="3D, C = 16, Sine error at half T"></td>
            <td><img src="readme_images/liif_compare_2d_3d_conv_enc/3d_16/half_H_mag.png" alt="3D, C = 16, Sine at half H"></td>
        </tr>
    </tbody>
</table>

<div align="center">
    <strong>3D convolutional encoder with sine activation LIIF is the best. Note that larger number of output channels for 2D convolution encoder results in static reconstruction.</strong>
</div>

### LIIF: Architecture Search
<table align="center" id="liif-architecture-search">
    <thead>
        <tr>
            <th>R</th>
            <th>Config</th>
            <th>Mag<span style="color: white;">.</span></th>
            <th>Mag<span style="color: white;">.</span> @ T / 2</th>
            <th>Mag<span style="color: white;">.</span> Error @ T / 2</th>
            <th>Mag<span style="color: white;">.</span> @ H / 2</th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td>3</td>
            <td>Original</td>
            <td><img src="readme_images/liif_architecture_search/R_3/original/mag.gif" alt="original"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/original/half_T_mag.png" alt="original at half T, R = 3"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/original/half_T_mag_error.png" alt="original, error at half T, R = 3"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/original/half_H_mag.png" alt="original at half H, R = 3"></td>
        </tr>
        <tr>
            <td>3</td>
            <td>ZF</td>
            <td><img src="readme_images/liif_architecture_search/R_3/ZF/mag.gif" alt="ZF"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/ZF/half_T_mag.png" alt="ZF at half T, R = 3"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/ZF/half_T_mag_error.png" alt="ZF, error at half T, R = 3"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/ZF/half_H_mag.png" alt="ZF at half H, R = 3"></td>
        </tr>
        <tr>
            <td>3</td>
            <td>UNet Encoder</td>
            <td><img src="readme_images/liif_architecture_search/R_3/UNet/mag.gif" alt="UNet Encoder"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/UNet/half_T_mag.png" alt="UNet Encoder at half T, R = 3"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/UNet/half_T_mag_error.png" alt="UNet Encoder, error at half T, R = 3"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/UNet/half_H_mag.png" alt="UNet Encoder at half H, R = 3"></td>
        </tr>
        <tr>
            <td>3</td>
            <td>RDN Encoder</td>
            <td><img src="readme_images/liif_architecture_search/R_3/RDN/mag.gif" alt="RDN Encoder"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/RDN/half_T_mag.png" alt="RDN Encoder at half T, R = 3"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/RDN/half_T_mag_error.png" alt="RDN Encoder, error at half T, R = 3"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/RDN/half_H_mag.png" alt="RDN Encoder at half H, R = 3"></td>
        </tr>
        <tr>
            <td>3</td>
            <td>Temporal TV</td>
            <td><img src="readme_images/liif_architecture_search/R_3/TV/mag.gif" alt="Temporal TV"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/TV/half_T_mag.png" alt="Temporal TV at half T, R = 3"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/TV/half_T_mag_error.png" alt="Temporal TV, error at half T, R = 3"></td>
            <td><img src="readme_images/liif_architecture_search/R_3/TV/half_H_mag.png" alt="Temporal TV at half H, R = 3"></td>
        </tr>
        <tr>
            <td>6</td>
            <td>Original</td>
            <td><img src="readme_images/liif_architecture_search/R_6/original/mag.gif" alt="original"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/original/half_T_mag.png" alt="original at half T, R = 6"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/original/half_T_mag_error.png" alt="original, error at half T, R = 6"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/original/half_H_mag.png" alt="original at half H, R = 6"></td>
        </tr>
        <tr>
            <td>6</td>
            <td>ZF</td>
            <td><img src="readme_images/liif_architecture_search/R_6/ZF/mag.gif" alt="ZF"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/ZF/half_T_mag.png" alt="ZF at half T, R = 6"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/ZF/half_T_mag_error.png" alt="ZF, error at half T, R = 6"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/ZF/half_H_mag.png" alt="ZF at half H, R = 6"></td>
        </tr>
        <tr>
            <td>6</td>
            <td>UNet Encoder</td>
            <td><img src="readme_images/liif_architecture_search/R_6/UNet/mag.gif" alt="UNet Encoder"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/UNet/half_T_mag.png" alt="UNet Encoder at half T, R = 6"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/UNet/half_T_mag_error.png" alt="UNet Encoder, error at half T, R = 6"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/UNet/half_H_mag.png" alt="UNet Encoder at half H, R = 6"></td>
        </tr>
        <tr>
            <td>6</td>
            <td>RDN Encoder</td>
            <td><img src="readme_images/liif_architecture_search/R_6/RDN/mag.gif" alt="RDN Encoder"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/RDN/half_T_mag.png" alt="RDN Encoder at half T, R = 6"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/RDN/half_T_mag_error.png" alt="RDN Encoder, error at half T, R = 6"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/RDN/half_H_mag.png" alt="RDN Encoder at half H, R = 6"></td>
        </tr>
        <tr>
            <td>6</td>
            <td>Temporal TV</td>
            <td><img src="readme_images/liif_architecture_search/R_6/TV/mag.gif" alt="Temporal TV"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/TV/half_T_mag.png" alt="Temporal TV at half T, R = 6"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/TV/half_T_mag_error.png" alt="Temporal TV, error at half T, R = 6"></td>
            <td><img src="readme_images/liif_architecture_search/R_6/TV/half_H_mag.png" alt="Temporal TV at half H, R = 6"></td>
        </tr>
    </tbody>
</table>

<div align="center">
    <strong>RDN encoder performs better than both shallow UNet encoder and temporal TV by large margin under both acceleration rates.</strong>
</div>
