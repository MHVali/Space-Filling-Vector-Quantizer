import torch
import torch.nn.functional as F
from utils import codebook_initialization
from torch.distributions import normal

class SpaceFillingVQ(torch.nn.Module):
    def __init__(self, desired_vq_bitrate, embedding_dim, device, initial_codebook=None, backpropagation=False):
        super().__init__()

        """
        Inputs:

        1. desired_vq_bitrate = Final bitrate desired for vector quantization

        2. embedding_dim = Embedding dimension (dimensionality of each input data sample or codebook entry)

        3. device = The device which executes the code (CPU or GPU)

        4. initial_codebook = Initial codebook entries to start training (shape: 4 x embedding_dim)
            "initial_codebook" has to contain only 4 entries, because SpaceFillingVQ optimization starts by only 4
            codebook entries (corner points), then it will be expanded to contain 2 ** desired_vq_bitrate entries.

        5. backpropagation = whether to pass gradients through SpaceFillingVQ to other trainable modules in the
        computational graph
        
        backpropagation = False : If we intend to train the SpaceFillingVQ module exclusively (independent from any 
        other module that requires training) on a distribution. In this case, we use mean squared error (MSE) between
        the input vector and its quantized version as the loss function (like in the "train.py").
        
        backpropagation = True : If we intend to train the SpaceFillingVQ jointly with other modules that requires
        gradients for training, we need to pass the gradients through SpaceFillingVQ module using
        "noise_substitution_vq" function. In this case, you do not need to define or add an exclusive loss term for
        SpaceFillingVQ optimization. The optimization loss function must only include the global loss function you 
        use for your specific application.
        """

        if initial_codebook is None:
            codebook = torch.zeros((int(2 ** desired_vq_bitrate), embedding_dim), device=device)
            codebook[0:4] = codebook_initialization(torch.randn(int(1e4), embedding_dim)).to(device)
        else:
            codebook = torch.zeros((int(2 ** desired_vq_bitrate), embedding_dim), device=device)
            codebook[0:4] = initial_codebook.to(device)

        self.embedding_dim = embedding_dim
        self.backpropagation = backpropagation
        self.device = device
        self.normal_dist = normal.Normal(0,1)
        self.eps = 1e-12

        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)


    def decode(self, fractional_index, codebook=None):
        if codebook is None:
            codebook = self.current_codebook

        entries = codebook.shape[0]
        integer_index = ((torch.floor(fractional_index)).clamp(min=0, max=entries - 2)).to(torch.int64)
        reminder_index = (fractional_index - integer_index).unsqueeze(-1)
        c0 = codebook[integer_index]
        c1 = codebook[integer_index + 1]
        return ((1 - reminder_index) * c0) + (reminder_index * c1)


    def forward(self, input_data, entries):
        """
        This function performs the main proposed space-filling vector quantization function.
        Use this forward function for training phase.

        N: number of input data samples
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        inputs:
                input_data: input data matrix which is going to be vector quantized | shape: (NxD)
                entries: current bitrate for space-filling vector quantizatizer : number of codebook entries
                contributing at this stage of training. In the initial stage, it starts by 4, and as the training
                continues, it will be expanded step by step (from this list [4, 8, 16, 32, 64, ...]) to reach the
                desired VQ bitrate. Take a look at "train.py" as the example.
        outputs:
                final_quantized_input: space-filling vector quantized version of input data | shape: (NxD)
                perplexity: average usage of codebook entries)
                integer_index: codebook indices selected for quantization in this forward pass
        """

        self.entries = entries
        self.current_codebook = self.codebook[0:self.entries]

        dither = torch.rand(self.entries-1, requires_grad=False, device=self.device)
        fractional_index = dither + torch.linspace(0, self.entries - 2, self.entries - 1, device=self.device)
        dithered_codebook = self.decode(fractional_index)

        distance = (input_data.unsqueeze(-1) - dithered_codebook.t().unsqueeze(0)).square().sum(dim=1)
        integer_index = distance.argmin(dim=1)

        quantized_input = self.decode(integer_index.to(torch.float), codebook=dithered_codebook)

        if self.backpropagation == True:
            final_quantized_input = self.noise_substitution_vq(input_data, quantized_input)
        else:
            final_quantized_input = quantized_input

        # Perplexity (average codebook usage)
        encodings = F.one_hot(integer_index, self.entries).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return final_quantized_input, perplexity.detach(), integer_index.cpu()


    def noise_substitution_vq(self, input_data, hard_quantized_input):
        random_vector = self.normal_dist.sample(input_data.shape).to(input_data.device)
        norm_hard_quantized_input = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
        vq_error = ((norm_hard_quantized_input / norm_random_vector + self.eps) * random_vector)
        quantized_input = input_data + vq_error
        return quantized_input


    def evaluation(self, input_data):
        """
        This function performs the space-filling vector quantization function for inference (evaluation) time.
        This function should not be used during training.

        N: number of input data samples
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input:
                input_data: input data matrix which is going to be space-filling vector quantized | shape: (NxD)
        output:
                quantized: space-filling vector quantized version of input data used for inference (eval) | shape: (NxD)
        """

        optimized_codebook = self.codebook.detach().clone()
        entries = optimized_codebook.shape[0]

        distance = (input_data.unsqueeze(-1) - optimized_codebook.t().unsqueeze(0)).square().sum(dim=1)
        integer_index = distance.argmin(dim=1).clamp(min=1, max=entries-2)

        cm = optimized_codebook[integer_index - 1]
        cc = optimized_codebook[integer_index]
        cp = optimized_codebook[integer_index + 1]

        reminder_index_m = (((cc - cm) * (input_data - cm)).sum(dim=1) / (cc - cm).square().sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)
        reminder_index_p = (((cp - cc) * (input_data - cc)).sum(dim=1) / (cp - cc).square().sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)

        xhat_m = ((1 - reminder_index_m) * cm) + (reminder_index_m * cc)
        xhat_p = ((1 - reminder_index_p) * cc) + (reminder_index_p * cp)
        distance_m = (input_data - xhat_m).square().sum(dim=1)
        distance_p = (input_data - xhat_p).square().sum(dim=1)

        offset = (distance_p < distance_m).to(torch.int).squeeze() - 1

        c0 = optimized_codebook[integer_index + offset]
        c1 = optimized_codebook[integer_index + offset + 1]

        reminder_index = (((c1 - c0) * (input_data - c0)).sum(dim=1) / (c1 - c0).square().sum(dim=1)).clamp(min=0., max=1.)

        quantized = c0 + (reminder_index.reshape(-1, 1) * (c1 - c0))

        return quantized