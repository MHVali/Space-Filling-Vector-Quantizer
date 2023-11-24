# **Space-Filling Vector Quantizer (SFVQ)**

This repository contains PyTorch implementation of the SFVQ technique. This technique combines two cocepts of vector quantization and space-filling curves such that for a given distrubition, SFVQ learns a piece-wise linear curve which fills an N dimensional distribution. We used this technqiue to make the latent space of a deep neural network more interpretable and we published it as a paper entitled "[Interpretable Latent Space Using Space-Filling Curves for Phonetics Analysis in Voice Conversion](https://www.isca-speech.org/archive/pdfs/interspeech_2023/vali23_interspeech.pdf)" in Interspeech conference 2023. SFVQ is a generic tool for modeling any distribution and it is neither restricted to any specific neural network architecture nor any data type (e.g. image, video, speech, etc.). Here is a visulization of how a Vector Quantizer(VQ) and a Space-Filling Vector Quantizer (SFVQ) model a 2D pentagon distribution.

![alt text](https://github.com/MHVali/Space-Filling-VQ/blob/main/vq_sfvq.jpg?raw=true)

# **Contents of this repository**

- `spacefilling_vq.py`: contains the main class of Space-Filling Vector Quantizer
- `train.py`: an example showing how to use and optimize Space-Filline Vector Quantizer to learn a Normal distribution
- `utils.py`: contains some utility functions used in other codes
- `plot_training_logs.py`: plots the training logs (which was saved druring execution of "train.py") in a pdf file  

Due to some limitations of TensorBoard, we prefered our own custom logging function (plot_training_logs.py).

# **Required packages**
- Python: (version 3.8 or higher)
- PyTorch (version: 1.13.0)
- Numpy (version: 1.22.2)
- Matplotlib (version: 3.6.2)

You can create the Python environment to run this project by passing the following lines of code in your terminal window in the following order:

`conda create --name sfvq python=3.8`  
`pip install torch==1.13.0`  
`pip install numpy==1.22.2`  
`pip install matplotlib==3.6.2`

You can also install all the above requirements by running the following command in your Python environment:  
`pip install -r requirments.txt`

# **Important note about training Space-Filling Vector Quantizer**

In the "spacefilling_vq.py" code, there is a boolean variable "backpropagation" which should be set based on one of the following situations:

- **backpropagation=False**: If we intend to train the SpaceFillingVQ module exclusively (independent from any other module that requires training) on a distribution. In this case, we use the mean squared error (MSE) between the input vector and its quantized version as the loss function (exactly like what we did in the "train.py").

- **backpropagation=True**: If we intend to train the SpaceFillingVQ jointly with other modules that requires gradients for training, we pass the gradients through the SpaceFillingVQ module using our recently porposed [Noise Substitution in Vector Quantization (NSVQ)](https://ieeexplore.ieee.org/abstract/document/9696322) technique. In this case, you do not need to define or add an exclusive loss term for SpaceFillingVQ optimization. The optimization loss function must only include the global loss function you use for your specific application.

# **Abstract of the paper**

Vector quantized variational autoencoders (VQ-VAE) are well-known deep generative models, which map input data to a latent space that is used for data generation. Such latent spaces are unstructured and can thus be difficult to interpret. Some earlier approaches have introduced a structure to the latent space through supervised learning by defining data labels as latent variables. In contrast, we propose an unsupervised technique incorporating space-filling curves into vector quantization (VQ), which yields an arranged form of latent vectors such that adjacent elements in the VQ codebook refer to similar content. We applied this technique to the latent codebook vectors of a VQ-VAE, which encode the phonetic information of a speech signal in a voice conversion task. Our experiments show there is a clear arrangement in latent vectors representing speech phones, which clarifies what phone each latent vector corresponds to and facilitates other detailed interpretations of latent vectors.

# **Cite the paper as**

Mohammad Hassan Vali and Tom Bäckström, “Interpretable Latent Space Using Space-Filling Curves for Phonetics Analysis in Voice Conversion”, in Proceedings of Interspeech, 2023.

```bibtex
@inproceedings{vali2023sfvq,
  title={{I}nterpretable {L}atent {S}pace {U}sing {S}pace-{F}illing {C}urves for {P}honetics {A}nalysis in {V}oice {C}onversion},
  author={Vali, Mohammad Hassan and Bäckström, Tom},
  booktitle={Proceedings of Interspeech},
  year={2023}
}
```
