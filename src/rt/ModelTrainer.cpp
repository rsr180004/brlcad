#include "ModelTrainer.h"
#include <string>
#include <fstream>
#include <iostream>
#include <Python.h>


/*
THIS IS THE CONSTRUCTOR FOR THE MODEL TRAINER CLASS. IT
EFFECTIVELY TRAINS THE 

*/
ModelTrainer::ModelTrainer(const char* db_path, const char* object_name) {
    // Initialize the Python interpreter
    Py_Initialize();

    // Add the current directory to sys.path
    PyRun_SimpleString("import sys");

    PyRun_SimpleString("sys.path.append('../src/rt')");

    std::string database_name = std::string(db_path).substr(0, std::string(db_path).length() - 2);
 
    std::string model_path = database_name + "_" + object_name;

    // Check if file exists in "./model_weights/"
    std::string file_path = std::string("./model_weights/") + database_name + "_" + object_name + "_weights.pth";

    std::ifstream file(file_path);

    /*

    --- CODE REQUIRED FOR SHADING MODEL

    // Check if file exists in "./model_weights/"
    std::string file_path_2 = std::string("./model_weights/") + database_name + "_" + object_name + "_hits_weights.pth";

    std::ifstream file2(file_path_2);
    */

     // Import our Python model module
    pName = PyUnicode_DecodeFSDefault("neural_trainer");
    pModule = PyImport_Import(pName);
    Py_XDECREF(pName);

    if(pModule != NULL) {

        pModelClass = PyObject_GetAttrString(pModule, "HitClassifier");
        if (PyCallable_Check(pModelClass)) {

            // If the model hasn't yet been trained (no weights saved)
            if(!file) {
                std::cout << "Model 1 has not yet been trained." << std::endl;
                PyObject* pTrainFunc = PyObject_GetAttrString(pModule, "train_model_1");
                // Create a tuple to hold the argument for train_model_1
                PyObject * pArgs = PyTuple_Pack(1, PyUnicode_FromString(model_path.c_str()));

                // Call function to train model
                PyObject_CallObject(pTrainFunc, pArgs);

                Py_XDECREF(pTrainFunc);
                Py_XDECREF(pArgs);
            } else {
                std::cout << "Model 1 has already been trained. Use existing model weights or delete file in model_weights directory to retrain." << std::endl; 
            }

            // Now there is a file with the model weights
            // Load or create the model instance
            PyObject* pArgs = PyTuple_Pack(1, PyUnicode_FromString(file_path.c_str()));
            pModelInstance = PyObject_CallObject(pModelClass, pArgs);

            Py_XDECREF(pArgs);
        }
        Py_XDECREF(pModelClass);

        /*

        ---- CODE FOR SHADING MODEL ---

        pModelClassReg = PyObject_GetAttrString(pModule, "Distance_Az_El_Model");
        if (PyCallable_Check(pModelClassReg)) {
            // If the model hasn't yet been trained (no weights saved)
            if(!file2) {
                std::cout << "Model 2 has not yet been trained." << std::endl;
                PyObject* pTrainFunc = PyObject_GetAttrString(pModule, "train_model_2");
                // Create a tuple to hold the argument for train_model_1
                PyObject * pArgs = PyTuple_Pack(1, PyUnicode_FromString(model_path.c_str()));
                PyObject_CallObject(pTrainFunc, pArgs);
                // printf("returned from train neural net");
                Py_XDECREF(pTrainFunc);
                Py_XDECREF(pArgs);
            } else {
                std::cout << "Model 2 has already been trained. Using existing model weights" << std::endl; 
            }

            // Now there is a file with the model weights
            // Load or create the model instance
            // Assuming the model's constructor takes the weights file path
            // printf("calling the model constructor\n");
            PyObject* pArgs = PyTuple_Pack(1, PyUnicode_FromString(file_path_2.c_str()));
            pModel2Instance = PyObject_CallObject(pModelClassReg, pArgs);

            Py_XDECREF(pArgs);
        }
        Py_XDECREF(pModelClassReg);
        */
    }
    Py_XDECREF(pModule);
}

ModelTrainer::~ModelTrainer() {
    // Cleanup
    Py_XDECREF(pModelInstance);
    /* Commented out below is for model 2*/
    // Py_XDECREF(pModel2Instance);
    Py_XDECREF(pModule);
    Py_Finalize();
}
