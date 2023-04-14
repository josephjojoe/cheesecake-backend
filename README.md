# Cheesecake
**Cheesecake** is a visual programming environment for AI model building, written in C#.

## Features
- **Customisable model architectures** - layer units, weight and bias initialisations, activations, etc
- **Choose your dataset!** It currently must be supplied in a specified format (as detailed in `Data.cs`), but work is ongoing to allow custom delimiters in text files
- **Model training** with different batch sizes, cost functions, learning rates, etc
- Support for **multi-branch layer topologies**
- **Parameter loading and saving**
- Written with **minimal/no use of external modules**

## Being worked on
- **GUI** - I will be using WPF for my GUI after I finish implementing the backend (not much more to go!).
- **Metrics** - it's fairly simple to implement basic metrics such as accuracy.

## Further development
- **More layer types**. This is my primary long-term ambition - I'm intending that Cheesecake becomes a tool for beginners to sharpen their intuition surrounding model building, which is so far only possible through the fully connected layers I have implemented. I'd like to implement recurrent (RNNs, LSTMs, GRUs) and CNN layers in the future to broaden use cases.
- **Support for more dataset types**. This links strongly to the previous aim - more layer types implies support for a wider range of datasets, such as audio or images.

## Miscellaneous
#### Why the name Cheesecake?
It's pretty sweet!
