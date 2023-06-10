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
Larger &lambda; results in smoother signals in temporal dimension.</strong>
</div>
