// ModelTrainer.h

#ifndef MODEL_TRAINER_H
#define MODEL_TRAINER_H

#include <Python.h>

class ModelTrainer {
public:
    // Constructor to initialize the neural ray tracer with a model path
    ModelTrainer(const char* model_path, const char* object_name);
    
    // Destructor to handle cleanup
    ~ModelTrainer();

private:
    PyObject *pName, *pModule, *pModelClass, *pModelClassReg, *pModelInstance, *pModel2Instance;
}; 

#endif // MODEL_TRAINER_H