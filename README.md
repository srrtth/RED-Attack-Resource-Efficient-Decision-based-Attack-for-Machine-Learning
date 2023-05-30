# RED-Attack-Resource-Efficient-Decision-based-Attack-for-Machine-Learning

This repository contains an implementation of the RED-Attack, a resource-efficient decision-based imperceptible attack method, as proposed in the research paper [RED-Attack: Resource Efficient Decision-based Imperceptible Attack for Machine Learning][https://arxiv.org/pdf/1901.10258.pdf]

The RED-Attack addresses the vulnerabilities of deep neural networks (DNNs), such as training or inference data poisoning, backdoors, Trojans, and model stealing. Unlike other black-box-based attacks that rely on output probability vectors/distributions, the RED-Attack conceals this information to mitigate such attacks.

## Motivation

Traditional decision-based attacks require a large number of queries to generate a single untargeted attack image, which is not suitable for resource-constrained systems where real-time performance is crucial. To overcome this limitation, the RED-Attack introduces a resource-efficient methodology to generate imperceptible attacks for black-box models.

## Methodology

The RED-Attack consists of two main steps: classification boundary estimation and adversarial noise optimization.

1. Classification Boundary Estimation:
   - The proposed half-interval search-based algorithm estimates a sample on the classification boundary using a target image and a randomly selected image from another class.

2. Adversarial Noise Optimization:
   - An optimization algorithm introduces small perturbations in randomly selected pixels of the estimated sample.
   - To ensure imperceptibility, the algorithm optimizes the distance between the perturbed sample and the target sample.

## Prerequisites

- Python 3.x
- numpy
- pandas
- matplotlib
- tensorflow
- cv2 (OpenCV)
- PIL (Python Imaging Library)

## Usage

1. Installation:
   - Clone the repository:

     ```shell
     git clone https://github.com/your-username/red_attack.git
     ```

   - Change the current directory to the cloned repository:

     ```shell
     cd red-attack-implementation
     ```

   - Install the required Python packages:

     ```shell
     pip install -r requirements.txt
     ```

Prepare the Dataset:

The input images should be organized in a directory structure where each subdirectory represents a class.
Update the path variable in the code to point to the directory containing the training images.
Ensure that the images are named appropriately and the class labels are assigned correctly.
Load the Pre-trained Model:
 
Update the path to the pre-trained model file (find it here: [https://github.com/srrtth/GTSRB-DNN-ImageClass]) in the code.
The model should be compatible with TensorFlow's Keras API.
Run the Adversarial Attack:

Update the paths to the test images (source_image_path and target_image_path) in the code.
Adjust the parameters for the attack (e.g., number of iterations, theta, j, dmin) as needed.
Execute the code to generate the adversarial examples and evaluate their effectiveness.

## Results

The RED-Attack implementation is evaluated on the CFAR-10 and German Traffic Sign Recognition (GTSR) datasets using state-of-the-art networks. The experimental results demonstrate that the RED-Attack generates adversarial examples with superior imperceptibility compared to traditional decision-based attacks, achieving significant improvements in perturbation norm, SSIM, and Correlation Coefficient metrics with a limited number of iterations (1000).

## Contributing

Contributions to this repository are welcome. If you find any issues or have improvements to suggest, please create a pull request or submit an issue.

## License

This project is licensed under the [MIT License](LICENSE).

---

Please note that this code is based on the paper "RED-Attack: Resource Efficient Decision-based Imperceptible Attack for Machine Learning" by Faiq Khalid, Hassan Ali, Muhammad Abdullah Hanif, Semeen Rehman, Rehan Ahmed, and Muhammad Shafique. The paper should be referred to for a more comprehensive understanding of the RED-Attack methodology and its technical details.
