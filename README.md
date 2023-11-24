# **Space-Filling Vector Quantizer (SFVQ)**

This repository contains PyTorch implementation of the SFVQ technique. This technique combines two cocepts of vector quantization and space-filling curves such that for a given distrubition, SFVQ learns a piece-wise linear curve which fills an N dimensional distribution. We used this technqiue to make the latent space of a deep neural network more interpretable and we published it as a paper entitled "Interpretable Latent Space Using Space-Filling Curves for Phonetics Analysis in Voice Conversion" in Interspeech conference 2023. SFVQ is a generic tool for modeling any distribution and it is neither restricted to any specific neural network architecture nor any data type (e.g. image, video, speech, etc.). Here is a visulization of how a Vector Quantizer(VQ) and a Space-Filling Vector Quantizer (SFVQ) model a 2D pentagon distribution.

![alt text](https://github.com/MHVali/Space-Filling-VQ/blob/main/aaa.jpg?raw=true)
