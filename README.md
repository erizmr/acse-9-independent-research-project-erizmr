# acse-9-independent-research-project-erizmr
acse-9-independent-research-project-erizmr created by GitHub Classroom


## Introduction
A network for end-to-end fluid flow inference called Particle-Net is presented in this project. The Particle-Net is able to infer the flow velocity field for shallow water problems from images of advected particles. Particle-Net outperforms the state-of-the-art particle image velocimetry (PIV) software PIVlab on synthetic data sets, both in terms of accuracy and efficiency, especially on extremely low-quality particle images. The root means square error and relative error of Particle-Net inference result are $60\%$ lower than that of PIVlab. In addition, even though Particle-Net was trained using only synthetic data, it still shows a promising ability to infer flow fields from real lab data even with sparse particles, although it does not outperform PIVlab on all lab data. Particle-Net processes images 25 times faster than PIVlab and is able to process nearly 72 pairs per second, which is quite promising for real-time fluid flow inference.

![decoder_inference_net](https://user-images.githubusercontent.com/33411325/64019939-55324900-cb28-11e9-998c-b445222c502b.png)


## Installation instructions


## Documentation

## Repo Structure

![Software_architecture (1)](https://user-images.githubusercontent.com/33411325/64020043-8b6fc880-cb28-11e9-92a6-d06e00fece57.png)


## Dependencies
To be able to run this software, the following packages and versions are required:




## Author and Course Information

Author： Mingrui Zhang

Github: @erizmr

Email: mingrui.zhang18@imperial.ac.uk

## License
This software is licensed under the MIT [license](https://github.com/msc-acse/acse-9-independent-research-project-erizmr/blob/master/License)
