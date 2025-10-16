# TriCRF Usage Examples and Sample Data

## Overview

This document provides comprehensive usage examples for all models in the TriCRF framework, including sample data formats, configuration files, and programming examples.

## Model Comparison

| Model | Best For | Features | Memory Usage | Speed |
|-------|----------|----------|--------------|-------|
| **MaxEnt** | Simple classification | No sequence dependencies | Low | Fastest |
| **CRF** | Linear sequence labeling | Sequential dependencies | Medium | Fast |
| **TriCRF1** | Hierarchical SLU | Topic + string features | High | Medium |
| **TriCRF2** | Hierarchical SLU | Topic + integer features | High | Medium |
| **TriCRF3** | Complex SLU | Topic + shared features | Highest | Slowest |

## Data Format Examples

### 1. MaxEnt Data Format

**File**: `maxent_train.txt`
```
NONE word=hello word-1=<s> word+1=world
GREETING word=world word-1=hello word+1=</s>

NONE word=book word-1=<s> word+1=a
BOOKING word=flight word-1=a word+2=tomorrow
```

### 2. CRF Data Format

**File**: `crf_train.txt`
```
NONE word=i word-1=<s> word+1=want word+2=to
NONE word=want word-1=i word+2=to word+3=book
NONE word=to word-1=want word+2=book word+3=a
BOOK word=book word-1=to word+2=a word+3=flight
NONE word=a word-1=book word+2=flight word+3=to
FLIGHT word=flight word-1=a word+2=to word+3=boston
NONE word=to word-1=flight word+2=boston word+3=</s>
CITY word=boston word-1=to word+2=</s>

NONE word=what word-1=<s> word+1=time word+2=is
NONE word=time word-1=what word+2=is word+3=it
NONE word=is word-1=time word+2=it word+3=</s>
NONE word=it word-1=is word+2=</s>
```

### 3. TriCRF Data Format (Hierarchical)

**File**: `tricrf_train.txt`
```
FLIGHT i want to book a flight to boston tomorrow
NONE word=i word-1=<s> word+1=want word+2=to
NONE word=want word-1=i word+2=to word+3=book
NONE word=to word-1=want word+2=book word+3=a
BOOK word=book word-1=to word+2=a word+3=flight
NONE word=a word-1=book word+2=flight word+3=to
FLIGHT word=flight word-1=a word+2=to word+3=boston
NONE word=to word-1=flight word+2=boston word+3=tomorrow
CITY word=boston word-1=to word+2=tomorrow word+3=</s>
DATE word=tomorrow word-1=boston word+1=</s>

HOTEL i need a hotel room in new york
NONE word=i word-1=<s> word+1=need word+2=a
NONE word=need word-1=i word+2=a word+3=hotel
NONE word=a word-1=need word+2=hotel word+3=room
ROOM word=hotel word-1=a word+2=room word+3=in
ROOM word=room word-1=hotel word+2=in word+3=new
NONE word=in word-1=room word+2=new word+3=york
CITY word=new word-1=in word+2=york word+3=</s>
CITY word=york word-1=new word+1=</s>
```

## Configuration Examples

### 1. MaxEnt Configuration

**File**: `maxent_config.cfg`
```
model_type = MaxEnt
mode = both
train_file = maxent_train.txt
test_file = maxent_test.txt
model_file = maxent_model.bin
estimation = LBFGS-L2
l2_prior = 2.0
iter = 100
log_file = maxent.log
log_mode = 2
```

### 2. TriCRF3 Configuration (Recommended)

**File**: `tricrf3_config.cfg`
```
model_type = TriCRF3
mode = both
train_file = tricrf_train.txt
test_file = tricrf_test.txt
model_file = tricrf3_model.bin
estimation = LBFGS-L2
l2_prior = 2.0
iter = 100
initialize = PL
initialize_iter = 30
log_file = tricrf3.log
log_mode = 2
```

### 3. Training Only Configuration

**File**: `train_only.cfg`
```
model_type = TriCRF3
mode = train
train_file = large_dataset.txt
model_file = production_model.bin
estimation = LBFGS-L2
l2_prior = 1.0
iter = 200
initialize = PL
initialize_iter = 50
log_file = training.log
log_mode = 3
```

### 4. Testing Only Configuration

**File**: `test_only.cfg`
```
model_type = TriCRF3
mode = test
test_file = test_data.txt
model_file = production_model.bin
output_file = predictions.txt
confidence = true
log_file = testing.log
log_mode = 1
```

## Programming Examples

### 1. Basic Training and Testing

```cpp
#include "TriCRF3.h"
#include "Utility.h"

int main() {
    // Create model with logger
    tricrf::Logger logger("training.log", 2);
    tricrf::TriCRF3 model(&logger);
    
    // Load training data
    model.readTrainData("train.txt");
    
    // Initialize model parameters
    model.initializeModel();
    
    // Train with L2 regularization
    model.train(100, 2.0, false);
    
    // Save trained model
    model.saveModel("model.bin");
    
    // Load model for testing
    model.clear();
    model.loadModel("model.bin");
    
    // Test on new data
    model.test("test.txt", "output.txt");
    
    return 0;
}
```

### 2. Cross-Validation Example

```cpp
#include "TriCRF3.h"
#include "Utility.h"
#include <vector>
#include <string>

void crossValidation(const std::string& dataFile, int folds) {
    // Load all data
    tricrf::TriCRF3 model;
    model.readTrainData(dataFile);
    
    // Split data into folds (simplified)
    std::vector<std::string> foldFiles;
    for (int i = 0; i < folds; ++i) {
        foldFiles.push_back("fold_" + std::to_string(i) + ".txt");
    }
    
    // Train and test each fold
    for (int fold = 0; fold < folds; ++fold) {
        tricrf::Logger logger("cv_fold_" + std::to_string(fold) + ".log", 2);
        tricrf::TriCRF3 cvModel(&logger);
        
        // Train on other folds
        // (Implementation depends on data splitting logic)
        
        cvModel.train(100, 2.0, false);
        cvModel.test(foldFiles[fold], "cv_output_" + std::to_string(fold) + ".txt");
    }
}
```

### 3. Model Comparison Example

```cpp
#include "MaxEnt.h"
#include "CRF.h"
#include "TriCRF3.h"
#include "Utility.h"

void compareModels(const std::string& trainFile, const std::string& testFile) {
    std::vector<std::pair<std::string, tricrf::MaxEnt*>> models;
    
    // Create different models
    models.push_back({"MaxEnt", new tricrf::MaxEnt()});
    models.push_back({"CRF", new tricrf::CRF()});
    models.push_back({"TriCRF3", new tricrf::TriCRF3()});
    
    for (auto& modelPair : models) {
        std::string name = modelPair.first;
        tricrf::MaxEnt* model = modelPair.second;
        
        std::cout << "Training " << name << "..." << std::endl;
        
        // Train model
        model->readTrainData(trainFile);
        model->initializeModel();
        model->train(100, 2.0, false);
        
        // Test model
        std::string outputFile = name + "_output.txt";
        model->test(testFile, outputFile);
        
        std::cout << name << " completed. Output saved to " << outputFile << std::endl;
        
        delete model;  // Clean up memory
    }
}
```

## Advanced Usage Examples

### 1. Custom Feature Engineering

```cpp
// Custom feature extraction example
tricrf::Event extractCustomFeatures(const std::vector<std::string>& tokens, size_t pos) {
    tricrf::Event event;
    
    // Basic word features
    event.obs.push_back({getFeatureId("word=" + tokens[pos]), 1.0});
    
    // Context features
    if (pos > 0) {
        event.obs.push_back({getFeatureId("word-1=" + tokens[pos-1]), 1.0});
    }
    if (pos < tokens.size() - 1) {
        event.obs.push_back({getFeatureId("word+1=" + tokens[pos+1]), 1.0});
    }
    
    // Position features
    event.obs.push_back({getFeatureId("pos=" + std::to_string(pos)), 1.0});
    
    // Length features
    event.obs.push_back({getFeatureId("len=" + std::to_string(tokens[pos].length())), 1.0});
    
    // Capitalization features
    if (std::isupper(tokens[pos][0])) {
        event.obs.push_back({getFeatureId("capitalized"), 1.0});
    }
    
    return event;
}
```

### 2. Model Ensemble

```cpp
class ModelEnsemble {
private:
    std::vector<tricrf::TriCRF3*> models;
    
public:
    void addModel(tricrf::TriCRF3* model) {
        models.push_back(model);
    }
    
    void predict(const std::string& testFile, const std::string& outputFile) {
        std::vector<std::vector<double>> predictions(models.size());
        
        // Get predictions from each model
        for (size_t i = 0; i < models.size(); ++i) {
            // Run prediction and collect confidence scores
            // (Implementation depends on confidence extraction)
        }
        
        // Combine predictions (e.g., majority voting or weighted average)
        // Save final predictions to outputFile
    }
};
```

### 3. Incremental Learning

```cpp
class IncrementalLearner {
private:
    tricrf::TriCRF3* model;
    std::string baseModelFile;
    
public:
    IncrementalLearner(const std::string& baseModel) : baseModelFile(baseModel) {
        model = new tricrf::TriCRF3();
        model->loadModel(baseModel);
    }
    
    void updateModel(const std::string& newDataFile, int iterations = 50) {
        // Load new training data
        model->readTrainData(newDataFile);
        
        // Continue training from existing model
        model->train(iterations, 1.0, false);  // Lower regularization for updates
        
        // Save updated model
        model->saveModel(baseModelFile + ".updated");
    }
};
```

## Performance Optimization Tips

### 1. Memory Management

```cpp
// Use RAII for automatic cleanup
class ModelManager {
private:
    std::unique_ptr<tricrf::TriCRF3> model;
    std::unique_ptr<tricrf::Logger> logger;
    
public:
    ModelManager() : model(std::make_unique<tricrf::TriCRF3>()),
                    logger(std::make_unique<tricrf::Logger>("app.log", 2)) {
        model->setLogger(logger.get());
    }
    
    // Automatic cleanup in destructor
};
```

### 2. Batch Processing

```cpp
void processBatch(const std::vector<std::string>& inputFiles, 
                  const std::string& modelFile) {
    tricrf::TriCRF3 model;
    model.loadModel(modelFile);
    
    for (const auto& inputFile : inputFiles) {
        std::string outputFile = inputFile + ".out";
        model.test(inputFile, outputFile);
        
        // Optional: Clear intermediate results to save memory
        // model.clearIntermediateResults();
    }
}
```

### 3. Configuration Management

```cpp
class ConfigManager {
private:
    tricrf::Configurator config;
    
public:
    ConfigManager(const std::string& configFile) : config(configFile) {}
    
    tricrf::MaxEnt* createModel() {
        std::string modelType = config.get("model_type");
        
        if (modelType == "MaxEnt") {
            return new tricrf::MaxEnt();
        } else if (modelType == "TriCRF3") {
            return new tricrf::TriCRF3();
        }
        // ... other models
        
        return nullptr;
    }
    
    std::string getModelFile() { return config.get("model_file"); }
    std::string getTrainFile() { return config.get("train_file"); }
    int getIterations() { return std::stoi(config.get("iter")); }
    double getRegularization() { return std::stod(config.get("l2_prior")); }
};
```

## Error Handling Examples

### 1. Robust Model Loading

```cpp
bool loadModelSafely(tricrf::MaxEnt* model, const std::string& modelFile) {
    try {
        if (!model->loadModel(modelFile)) {
            std::cerr << "Failed to load model from " << modelFile << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception while loading model: " << e.what() << std::endl;
        return false;
    }
}
```

### 2. Data Validation

```cpp
bool validateDataFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    size_t lineCount = 0;
    while (std::getline(file, line)) {
        ++lineCount;
        
        // Basic validation
        if (line.empty()) continue;
        
        std::vector<std::string> tokens = tricrf::tokenize(line);
        if (tokens.empty()) {
            std::cerr << "Empty line at " << lineCount << std::endl;
            return false;
        }
    }
    
    std::cout << "Data validation passed. " << lineCount << " lines processed." << std::endl;
    return true;
}
```

## Sample Datasets

### 1. Airline Travel Information (ATIS)

**Domain**: Flight booking and information
**Labels**: FLIGHT, FROMLOC, TOLOC, DATE, TIME, etc.

```
FLIGHT show me flights from boston to denver
NONE word=show word-1=<s> word+1=me word+2=flights
NONE word=me word-1=show word+2=flights word+3=from
FLIGHT word=flights word-1=me word+2=from word+3=boston
FROMLOC word=from word-1=flights word+2=boston word+3=to
CITY word=boston word-1=from word+2=to word+3=denver
TOLOC word=to word-1=boston word+2=denver word+3=</s>
CITY word=denver word-1=to word+1=</s>
```

### 2. Hotel Booking

**Domain**: Hotel reservations
**Labels**: HOTEL, CITY, DATE, ROOM, etc.

```
HOTEL i need a hotel room in chicago for next week
NONE word=i word-1=<s> word+1=need word+2=a
NONE word=need word-1=i word+2=a word+3=hotel
NONE word=a word-1=need word+2=hotel word+3=room
ROOM word=hotel word-1=a word+2=room word+3=in
ROOM word=room word-1=hotel word+2=in word+3=chicago
NONE word=in word-1=room word+2=chicago word+3=for
CITY word=chicago word-1=in word+2=for word+3=next
NONE word=for word-1=chicago word+2=next word+3=week
DATE word=next word-1=for word+2=week word+3=</s>
DATE word=week word-1=next word+1=</s>
```

### 3. Restaurant Information

**Domain**: Restaurant queries
**Labels**: RESTAURANT, CUISINE, LOCATION, PRICE, etc.

```
RESTAURANT find me a chinese restaurant near downtown
NONE word=find word-1=<s> word+1=me word+2=a
NONE word=me word-1=find word+2=a word+3=chinese
NONE word=a word-1=me word+2=chinese word+3=restaurant
CUISINE word=chinese word-1=a word+2=restaurant word+3=near
RESTAURANT word=restaurant word-1=chinese word+2=near word+3=downtown
NONE word=near word-1=restaurant word+2=downtown word+3=</s>
LOCATION word=downtown word-1=near word+1=</s>
```

## Troubleshooting Guide

### Common Issues

1. **Memory Errors**
   - Reduce batch size or use smaller datasets
   - Monitor memory usage during training
   - Use `delete[]` for arrays allocated with `new[]`

2. **Training Convergence Issues**
   - Increase number of iterations
   - Adjust regularization parameters
   - Check data quality and format

3. **Low Accuracy**
   - Increase training data
   - Add more features
   - Try different model types
   - Adjust hyperparameters

4. **Slow Training**
   - Use smaller datasets for development
   - Reduce feature set size
   - Use faster models (MaxEnt, CRF) for initial experiments

### Performance Monitoring

```cpp
void monitorTraining(tricrf::TriCRF3* model, const std::string& logFile) {
    tricrf::Logger logger(logFile, 3);  // Verbose logging
    model->setLogger(&logger);
    
    // Training with progress monitoring
    model->train(100, 2.0, false);
    
    // Memory usage reporting
    // (Implementation depends on system-specific memory monitoring)
}
```

This comprehensive guide should help users effectively utilize the TriCRF framework for their specific needs.
