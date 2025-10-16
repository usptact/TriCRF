# TriCRF - Triangular-chain Conditional Random Fields

C++ Implementation of Triangular-chain CRFs for Spoken Language Understanding and Sequential Labeling

## Overview

TriCRF is a collection of machine learning models for sequential labeling tasks, particularly designed for spoken language understanding (SLU). The framework implements five different models ranging from simple maximum entropy to complex triangular-chain conditional random fields.

## Models

The TriCRF framework provides five distinct models, each with different capabilities and use cases:

### 1. Maximum Entropy (MaxEnt)
**Type**: Non-sequential probabilistic model  
**Inheritance**: Base class for all other models  
**Use Case**: Simple classification without sequence dependencies

**Description**: 
- A maximum entropy (log-linear) model for individual event classification
- No sequential dependencies between labels
- Each observation is classified independently
- Fastest training and inference
- Suitable for tasks where sequence structure is not important

**Data Structure**: Uses `Sequence` (vector of `Event`)

### 2. Linear-chain CRF (CRF)
**Type**: Sequential probabilistic model  
**Inheritance**: Extends MaxEnt  
**Use Case**: Standard sequential labeling with linear dependencies

**Description**:
- Traditional linear-chain conditional random field
- Models dependencies between adjacent labels in a sequence
- Uses forward-backward algorithm for inference
- Viterbi algorithm for finding the best label sequence
- Standard approach for named entity recognition, part-of-speech tagging

**Data Structure**: Uses `Sequence` (vector of `Event`)  
**Key Features**:
- Transition matrices (M) for label-to-label dependencies
- Observation matrices (R) for label-to-feature relationships
- Alpha/Beta matrices for forward-backward computation

### 3. Triangular-chain CRF Model 1 (TriCRF1)
**Type**: Hierarchical sequential model  
**Inheritance**: Extends CRF  
**Use Case**: Two-level hierarchical labeling with topic and sequence labels

**Description**:
- First triangular-chain model with hierarchical structure
- Models both topic-level and sequence-level dependencies
- Topic represents high-level semantic meaning (e.g., dialogue act)
- Sequence represents fine-grained labels within the topic
- Each topic has its own set of sequence parameters

**Data Structure**: Uses `TriStringSequence` (topic + string sequence)  
**Key Features**:
- Topic parameter (`m_ParamTopic`) for topic classification
- Multiple sequence parameters (`m_ParamSeq`) - one per topic
- Gamma matrix for topic prior probabilities
- Z matrix for topic normalization

### 4. Triangular-chain CRF Model 2 (TriCRF2)
**Type**: Optimized hierarchical sequential model  
**Inheritance**: Extends CRF  
**Use Case**: More efficient version of hierarchical labeling

**Description**:
- Optimized version of triangular-chain model
- Improved computational efficiency through better indexing
- Similar hierarchical structure to TriCRF1 but with performance optimizations
- Uses integer-based features instead of string features

**Data Structure**: Uses `TriSequence` (topic + integer sequence)  
**Key Features**:
- Speed optimizations with `m_zy_index` and `m_yz_index`
- Single sequence parameter (`m_ParamSeq`) shared across topics
- Single topic parameter (`m_ParamTopic`)
- More efficient memory usage

### 5. Triangular-chain CRF Model 3 (TriCRF3)
**Type**: Advanced hierarchical sequential model  
**Inheritance**: Extends CRF  
**Use Case**: Most sophisticated model with full feature support

**Description**:
- Most advanced triangular-chain model
- Combines benefits of TriCRF1 and TriCRF2
- Full string feature support with hierarchical structure
- Best performance for complex SLU tasks
- Used in the original research papers

**Data Structure**: Uses `TriStringSequence` (topic + string sequence)  
**Key Features**:
- Topic parameter (`m_ParamTopic`) for topic classification
- Multiple sequence parameters (`m_ParamSeq`) - one per topic
- Full string feature support
- Most comprehensive model architecture

## Model Selection Guide

| Model | Use When | Pros | Cons |
|-------|----------|------|------|
| **MaxEnt** | Simple classification, no sequence dependencies | Fast, simple | No sequential modeling |
| **CRF** | Standard sequential labeling | Well-established, good performance | No hierarchical structure |
| **TriCRF1** | Hierarchical labeling with string features | Full feature support | Slower than TriCRF2 |
| **TriCRF2** | Hierarchical labeling, performance critical | Fast, efficient | Integer features only |
| **TriCRF3** | Complex SLU tasks, best accuracy needed | Most sophisticated, best performance | Most complex, slower |

## Data Format

All models use a consistent data format where:

- **Examples are separated by blank lines**
- **First column**: Class label
- **For triangular models**: First row contains topic information
- **Subsequent rows**: Sequential labeling data

### Example Data Structure:
```
TOPIC_LABEL topic features...
SEQUENCE_LABEL word features...
SEQUENCE_LABEL word features...
SEQUENCE_LABEL word features...

TOPIC_LABEL topic features...
SEQUENCE_LABEL word features...
...
```

## Configuration

Models are selected via the configuration file:

```ini
model_type = TriCRF3  # {MaxEnt, CRF, TriCRF1, TriCRF2, TriCRF3}
mode = both           # {train, test, both}
train_file = data.txt
test_file = test.txt
model_file = model.bin
```

## Research Background

This implementation is based on research in spoken language understanding:

- **TriCRF Paper**: "Triangular-chain Conditional Random Fields" (Jeong & Lee, IEEE TASLP 2008)
- **Transfer Learning Paper**: "Multi-domain Spoken Language Understanding with Transfer Learning" (Jeong & Lee, Speech Communication 2009)

The triangular structure models the hierarchical nature of spoken language where:
- **Topic level**: Dialogue act or semantic intent (e.g., "FLIGHT", "HOTEL")
- **Sequence level**: Named entities and slot filling (e.g., "CITY_NAME-B", "DATE-B")

## Build and Usage

### Building
```bash
# Using CMake (recommended)
mkdir build && cd build
cmake ..
make -j4

# Using the build script
./build.sh
```

### Running
```bash
# Show usage
./build/bin/tricrf

# Train and test with configuration file
./build/bin/tricrf example/example.cfg
```

## Technical Details

### Key Algorithms
- **Parameter Estimation**: L-BFGS optimization with L1/L2 regularization
- **Inference**: Forward-backward algorithm for probability calculation
- **Decoding**: Viterbi algorithm for best sequence finding
- **Initialization**: Pseudolikelihood for faster convergence

### Performance Considerations
- **TriCRF2**: Fastest hierarchical model
- **TriCRF3**: Most accurate but computationally intensive
- **Memory Usage**: Increases with model complexity and feature count
- **Training Time**: Depends on data size, feature count, and iterations

## License

Modified BSD License - see LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{jeong2008triangular,
  title={Triangular-chain Conditional Random Fields},
  author={Jeong, Minwoo and Lee, Gary Geunbae},
  journal={IEEE Transactions on Audio, Speech, and Language Processing},
  year={2008}
}
```
