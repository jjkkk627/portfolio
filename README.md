# Portfolio

Hi! I'm currently pursuing an MPhil in Data Intensive Science at Cambridge, with a focus on machine learning, computer vision, and medical imaging. This portfolio highlights some of the projects I’ve worked on—ranging from foundational ML to applied medical imaging and advanced statistical methods.

---

## dual_autodiff Python package — [View Repo »]( https://github.com/jjkkk627/autodiff-with-dual-numbers.git)

A Python package implementing forward-mode automatic differentiation using dual numbers, developed as part of a university assignment. The package supports elementary functions, operator overloading, and symbolic gradient computation via dual number algebra.

### Features:
- Core `Dual` class implementing real + dual number arithmetic  
- Support for common mathematical functions: `sin`, `cos`, `log`, `exp`, etc.  
- Test suite using `pytest`  
- Documentation with Sphinx, including a tutorial notebook  
- Cythonized version for performance benchmarking  
- Comparison with numerical and analytical gradients  
- Built Linux wheels using `cibuildwheel` and Docker

### What I learned:
- The mathematical foundations of dual numbers and forward-mode autodiff  
- How to design and package a Python library from scratch, with proper structure and installation via `pyproject.toml`  
- Performance tradeoffs between pure Python and Cython implementations  
- How to build, document, test, and distribute a fully-functional Python package  
- Practical experience with low-level deployment tools like `cibuildwheel`, Docker, and wheel packaging
---

## MNIST Classifier — [View Repo »]( https://github.com/jjkkk627/machine-learning-mnist.git)

A project focused on learning to **sum two handwritten MNIST digits** using machine learning models. Each input consists of two vertically stacked 28×28 images (total 56×28), and the output is the sum of the digit labels (e.g., input = [4, 1] → target = 5).

### Features:
- Custom dataset builder to combine and label pairs of MNIST images  
- Fully connected neural network (regressor) trained to predict the sum  
- Hyperparameter tuning (activation function, learning rate, batch size, number of layers, and hidden units)  
- Comparison with other inference models:
  - Random Forests
  - Support Vector Machines
- t-SNE visualizations on learned embedding space vs raw input  
- Probability comparison using a weak linear classifier trained on combined vs sequentially applied images

### What I learned:
- How to generate new tasks and structured datasets from standard datasets (e.g., MNIST → addition task)  
- Tradeoffs between different model families (linear classifiers vs deep learning)  
- How data representations evolve inside a trained neural network (via t-SNE)  
- The importance of architectural choices, dataset splitting, and reproducibility in experimental ML work  
- Practical skills in scikit-learn and dimensionality reduction for model analysis
---

## 🎙️ Voice Flashcard Trainer (Hackathon Project) — [View Repo »](https://github.com/yourusername/voice-flashcard-trainer)

Built in a 6-hour hackathon: a voice-based flashcard trainer using real-time speech input, LLMs, and audio feedback. Created using the [Pyneuphonic](https://github.com/neuphonic/pyneuphonic) API.

**Features:**
- Asks flashcard questions aloud
- Listens to spoken answers
- Evaluates correctness and gives feedback
- Logs performance + ends when user says "stop"

---

## Medical Imaging Projects *(Private)*

I’ve completed coursework on:
- 3D CT segmentation using U-Net-style architectures
- Multi-coil MRI denoising using k-space data

Due to dataset size and privacy restrictions, these projects aren’t publicly available, but I’d be happy to share code structure or logic privately.

---

## What's Next

This portfolio is growing as I explore more:
- A Bayesian inference project modeling features of the Antikythera mechanism
- Interpretability work on CNNs for chest X-rays

Thanks for stopping by!

