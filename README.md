# ACSE-9 Independent Research Project
## Shallow water flow field inference using convolutional neuralnetworks with particle images
### @Mingrui Zhang


## Introduction
A network for end-to-end fluid flow inference called Particle-Net is presented in this project. The Particle-Net is able to infer the flow velocity field for shallow water problems from images of advected particles. Particle-Net outperforms the state-of-the-art particle image velocimetry (PIV) software PIVlab on synthetic data sets, both in terms of accuracy and efficiency, especially on extremely low-quality particle images. The root means square error and relative error of Particle-Net inference result are 60% lower than that of PIVlab. In addition, even though Particle-Net was trained using only synthetic data, it still shows a promising ability to infer flow fields from real lab data even with sparse particles, although it does not outperform PIVlab on all lab data. Particle-Net processes images 25 times faster than PIVlab and is able to process nearly 72 pairs per second, which is quite promising for real-time fluid flow inference.

<p align="center">
  <img src="https://user-images.githubusercontent.com/33411325/64025420-8e70b600-cb34-11e9-9545-1f279c2a12a1.png" width="619" height="420"><br>
</p>


## Sample results

- Particle Images

High quality input (particle density 0.07 particle per pixel)
<p align="center">
  <img src="https://user-images.githubusercontent.com/33411325/64683586-745f9d80-d47b-11e9-95d5-be2fb40184b9.gif" ><br>
</p>

Low quality input (particle density 0.007 particle per pixel)
<p align="center">
  <img src="https://user-images.githubusercontent.com/33411325/64683547-5eea7380-d47b-11e9-813f-63d598bb9552.gif" ><br>
</p>


- Inference results on unseen data with low quality input
![Re3450_veloc_200_predictions](https://user-images.githubusercontent.com/33411325/64683051-71b07880-d47a-11e9-8d81-d7be31c3acb2.gif)

- Ground Truth
![labels](https://user-images.githubusercontent.com/33411325/64683322-f4d1ce80-d47a-11e9-9646-24fd61fd8af8.gif)

## Installation instructions

- Firedrake and Thetis

The Thetis official installation instructions is [here](https://thetisproject.org/download.html)
Thetis requires installation of Firedrake (available for Ubuntu, Mac, and in principle other Linux and Linux-like systems) and must be run from within the Firedrake virtual environment. You can install both Firedrake and Thetis by running:

```bash
  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
  python3 firedrake-install --install thetis
```

- ParticleModule
The ParicleModule could be run on a virtual machine, for detail information and the authoration, please refer https://github.com/jrper/ParticleModule.


- Other package and libraries
It is recommended to use the command below to install other package and libraries.
```bash
   pip install numpy
   pip install scipy
   pip install sklearn 
   pip install matplotlib
   pip install torch
   pip install cmocean
   pip install vtk
```

## Documentation

The documentation of some core modules are generated by Pydoc and the export files are included in the doc folder.

## Repo Structure

<p align="center">
  <img src="https://user-images.githubusercontent.com/33411325/64020043-8b6fc880-cb28-11e9-92a6-d06e00fece57.png" width="484" height="214"><br>
</p>

## Run test cases

- Machine learning part
Although the codes in this part are designed for GPU, it can also be executed on a CPU only machine.
You can run a test train on a super mini size dataset:
```bash
  cd machine_learning
  python3 train.py
```
If the ram is not run out, you can see the moving training and validation loss.

Also a trained model is provided [here](https://drive.google.com/file/d/1ZD_XnRa3UW4NaDCQVukkvrS2EmHrW7Rb/view?usp=sharing). It is too large (about 443 mb)to put it online, so only the download link is posted here. It is recommended to download it and put it in the ```model_trained``` folder, then run the commands below.

```bash
  cd machine_learning
  python3 load_and_visualization.py
```
If you successfully download and put it in the folder ```model_trained``` and then load it into your machine. You can see some fluid flow velocity field images infered by the ParticleNet.

- Data processing

The DataProcessing notebook in the notebook folder is a collection of all data processing methods in this project. The data foramt includes vtu, vtp, txt, tif/png/jpg. You can play with some samples data by openning the notebook in its folder, and then click ```run all cells```.

## Dependencies
To be able to run this software, the following packages and versions are required:

- Firedrake (2019)
- Thetis (2019)
- Pytorch (1.1.0)
- ParticleModule (2019)
- SciPy (v1.3.0)
- NumPy (v1.16.4)
- vtk (v6.0.0)

## PowerPoint
The presentation PowerPoint download link is [here](https://github.com/msc-acse/acse-9-independent-research-project-erizmr/blob/master/final_presentation_Mingrui_Zhang.pptx?raw=true).

## Author and Course Information

Author： Mingrui Zhang

CID: 01535292

Github: @erizmr

Email: mingrui.zhang18@imperial.ac.uk

## License
This software is licensed under the MIT [license](https://github.com/msc-acse/acse-9-independent-research-project-erizmr/blob/master/License)
