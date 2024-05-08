# Fractal-Head Attention (FHA)
## about
This is my first branching-off from my new [customGPT]() repo meant mainly as a test of that repo since i don't expect this idea to perform that well. In it i'll be saying fuck it to efficiency & ram and giving each attention block heads of a variety of sizes in a kind of self-similar manner

This repo is part of a larger project of mine called [micro_model_sandbox]() that's basically a hub for all the novel model experiments I do, with the goal of facilitating easy comparison between the different models. Basically for each of those experiments I just use the [customGPT]() repo as a template to start editing, and then once I'm happy with the project (or if I've just abandoned it but it's moderately functional) I add it to the sandbox. If you end up using that repo as a template, as i did here for FHA, feel free to contribute your project to the sandbox as well!

## file structure
- `modules/`: where all of the code for the actual model goes
    - `fha.py`: This is the primary file that makes this project unique from [customGPT](). At some point I'll update this readme or maybe add a jupyter notebook that'll provide a full detailed visual walkthrough of the edit i've made to the traditional multi-query attention mechanism. 
    - `layer.py`: defines each residual connection layer of our GPT
    - `logging.py`: defines the `LoggingModule` class, a wrapper that you should use instead of pytorch's `nn.module` in order to facilitate easy demonstration of how tensor shapes change throughout a given module
    - `mlp.py`: a two-layer multi-layer perceptron with an optional gate and either [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [GeLU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html), or [SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) nonlinearities, all configurable in `config.py`. Adding more nonlinearities is also absurdly easy
    - `model.py`: the primary class for our GPT
    - `norm.py`: a norm module with an optional affine layer that allows you to switch between [RMSNorm](https://arxiv.org/abs/1910.07467), [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) and [CosineNorm](https://arxiv.org/pdf/1702.05870) easily using a setting over in `config.py`. Adding different normalization methods is also absurdly easy
- `tokenizers/bpe/`
    - `models/`
        - `{95, 128, 256, 512, 1024, 2048}.model`: the 95 one is character-wise tokenization. All others are [byte-pair encoding](https://huggingface.co/learn/nlp-course/en/chapter6/5), except instead of bytes i use the 95 unique characters that show up in the TinyStories dataset
    - `build.ipynb`: the notebook where i built my bpe tokenizers. My pairing rules could certainly be improved upon
    - `tokenizer.py`: an overly-simplistic and annoyingly inefficient tokenizer with bos & eos tokens, post-sequence padding, and a `display` function to help you visualize how a given string is broken down
- `trained/`
    - `FHA_GPT_{0.3m_2024-05-07|13-05-29, 0.8m_2024-05-05|10-54-35}/`: a series of tiny models designed to be compared against one another. they're not large enough to get intelligible output; planning to make some bigger ones and train them for longer at some point
        - `log_data.csv`: record of loss & perplexity data over the course of training
        - `model_config.json`: hyperparameters of the model
        - `model.pth`: weights of the model
        - `train_config.json`: hyperparameters of the training loop used to train the model
- `inference.ipynb`: open this notebook if you just want to see the output of one of the models
- `model_comparison.ipynb`: open this notebook to compare different models against each other. includes loss curve plots and topk teacher-forcing accuracy rate
- `testing_modules.ipynb`: creates easy printouts that allow you to follow the progression of tensor shapes for demonstration & debugging purposes of all the modules in `model.py`. If you're building new modules for a novel architecture idea you have then this notebook will be of extreme value to you in debugging & visualization
- `train.ipynb`: open this notebook to train a new model
- `config.py`: all of the editable model and training settings
- `inference.py`: functions for performing inference, used in `inference.ipynb` and `train.ipynb`
- `requirements.txt` - I should probably change this to only include the packages that are actually necessary and not be so strict on versions. The command I used to get this list is `pip freeze | grep -v " @ file://" > requirements.txt`, lmk if you know of a better method
- `tools.py`: A variety of functions & classes that don't fit elsewhere and/or are used by more than one of the jupyter notebooks
- `train.py`: functions for training a model, used in `train.ipynb`

## definite TODOs
- [ ] train some larger models for a full 1000 iterations that i can then compare with [customGPT]() ones over in [micro_model_sandbox]()

### potential future TODOs
- [ ] see if i can figure out how to make it into efficient tensor operations instead of for loops

## how to contribute
Other than the above TODO lists, appreciated contributions include:
- bug fixes
- adding more detailed comment explanations of what the code is doing
- general readability edits
- efficiency edits
- editing the code in `modules/` to take better advantage of the `LoggingModule`. This means splitting up each class into more and tinier functions
- training more models (especially if they're bigger than what's already here!)

Because I'm not super knowledgeable on how collaborating on git projects works and I tend to edit directly on the main branch, please reach out and communicate with me about any edits you plan to make so that I can avoid editing the same files. [Click here to join my discord server](https://discord.gg/hTYQyDPpr9)

## check me out
- guides on how to build miniature versions of popular models from scratch, with a hand-holding walkthrough of every single tensor operation: [minGemma](https://github.com/evintunador/minGemma), [minGrok](https://github.com/evintunador/minGrok), and [minLlama3](https://github.com/evintunador/minLlama3)
- [my YouTube channel](https://www.youtube.com/@Tunadorable)
- my [other links](https://linktr.ee/tunadorable)