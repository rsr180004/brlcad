#include "NeuralRayTracer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>

#include <pthread.h>
pthread_mutex_t model_mutex = PTHREAD_MUTEX_INITIALIZER;

NeuralRayTracer* NeuralRayTracer_Create(const char* model_path) {

    NeuralRayTracer* instance = malloc(sizeof(NeuralRayTracer));
    

    PyEval_InitThreads();
    Py_Initialize();

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../src/rt')");

    char file_path[256];
    snprintf(file_path, sizeof(file_path), "model_weights/%s_weights.pth", model_path);
    FILE* file = fopen(file_path, "r");
    /*
    --- THIS CODE IS USED FOR THE SHADING MODEL

    char file_path_2[256];
    snprintf(file_path_2, sizeof(file_path_2), "model_weights/%s_hits_weights.pth", model_path);
    FILE* file2 = fopen(file_path_2, "r");
    */

    instance->pName = PyUnicode_DecodeFSDefault("neural_trainer");

    instance->pModule = PyImport_Import(instance->pName);
    Py_XDECREF(instance->pName);

    if(instance->pModule != NULL) {

        instance->pModelClass = PyObject_GetAttrString(instance->pModule, "HitClassifier");
        if (PyCallable_Check(instance->pModelClass)) {  

            // This should never be the case (model should already get trained)
            if(!file) {
                printf("Can't locate model #1 for this specific database and object.\n");
                exit(EXIT_FAILURE);
            } else {
                printf("A HitClassifier Model Exists!\n");

                // Instantiate first
                PyObject* pArgs = PyTuple_Pack(1, PyLong_FromLong(5)); // Assuming input size is 5

                instance->pModelInstance = PyObject_CallObject(instance->pModelClass, pArgs);

                Py_XDECREF(pArgs);

                printf("The model is going to be trained on the loaded weights\n");

                PyObject *pLoadWeightsMethod = PyObject_GetAttrString(instance->pModelInstance, "load_model1_weights");
                if (PyCallable_Check(pLoadWeightsMethod)) {
                    //printf("callable\n");
                    PyObject *p_newArgs = PyTuple_Pack(1, PyUnicode_FromString(file_path));
                    PyObject_CallObject(pLoadWeightsMethod, p_newArgs);
                    Py_XDECREF(p_newArgs);
                } 
                Py_XDECREF(pLoadWeightsMethod);

                printf("Model #1 is now trained on the loaded weights\n");
                fclose(file);
            }
        }

        /*

        --- THIS CODE IS FOR THE SHADING MODEL

        instance->pModelClassReg = PyObject_GetAttrString(instance->pModule, "Distance_Az_El_Model");
        if (PyCallable_Check(instance->pModelClassReg)) {  
            printf("in class\n");
            // This should never be the case (model should already get trained)
            if(!file2) {
                printf("Can't locate model #2 for this specific database and object.\n");
                exit(EXIT_FAILURE);
            } else {
                printf("theres a model 2 alive\n");

                // Instantiate CustomResNet first
                PyObject* pArgs = PyTuple_Pack(1, PyLong_FromLong(5)); // Assuming input size is 5
                //printf("calling init\n");
                instance->pModel2Instance = PyObject_CallObject(instance->pModelClassReg, pArgs);


                //printf("pModelInstance in creator: %p\n", (void*)instance->pModelInstance);
                //printf("called init\n");
                Py_XDECREF(pArgs);

                printf("Model 2 is going to be trained on the loaded weights\n");

                // Print the weights before and after loading weights
                //printf("WEIGHTS BEFORE\n");

                PyObject *pLoadWeightsMethod = PyObject_GetAttrString(instance->pModel2Instance, "load_model2_weights");
                if (PyCallable_Check(pLoadWeightsMethod)) {
                    //printf("callable\n");
                    PyObject *p_newArgs = PyTuple_Pack(1, PyUnicode_FromString(file_path_2));
                    PyObject_CallObject(pLoadWeightsMethod, p_newArgs);
                    Py_XDECREF(p_newArgs);
                } 
                Py_XDECREF(pLoadWeightsMethod);

                printf("The models are now trained on the loaded weights\n");
                fclose(file2);
            }
        }
        */
        //Py_XDECREF(instance->pModelClass); --> This caused the issue of the function not being callable in the shoot ray
        
    }
    Py_XDECREF(instance->pModule);
    

    return instance;
     
}

void NeuralRayTracer_Destroy(NeuralRayTracer* instance) {
    Py_XDECREF(instance->pModelInstance);
    /* THE FOLLOWING LINES ARE COMMENTED OUT. UNCOMMENT IF ENABLING SHADING MODEL */
    // Py_XDECREF(instance->pModel2Instance);
    Py_XDECREF(instance->pModelClass);
    // Py_XDECREF(instance->pModelClassReg);
    Py_Finalize();
    pthread_mutex_destroy(&model_mutex);
    free(instance);
}

void NeuralRayTracer_ShootRay(NeuralRayTracer* instance, double origin[3], double az_el[2], double output[1]) {
    
    // Lock the mutex to ensure exclusive access to the model
    pthread_mutex_lock(&model_mutex);

    if (instance->pModelInstance != NULL) {

        int present = PyObject_HasAttrString(instance->pModelInstance, "predict_hit_or_miss");

        PyObject* pMethod = PyObject_GetAttrString(instance->pModelInstance, "predict_hit_or_miss");
        if (PyCallable_Check(pMethod)) {

            PyObject* pArgs = PyTuple_Pack(2, 
                                           Py_BuildValue("[d,d,d]", origin[0], origin[1], origin[2]),
                                           Py_BuildValue("[d,d]", az_el[0], az_el[1]));
            
            PyObject* pResult = PyObject_CallObject(pMethod, pArgs);

            
            if (PyList_Check(pResult) && PyList_Size(pResult) == 1) {

                for (int i = 0; i < 1; i++) {

                    PyObject* pItem = PyList_GetItem(pResult, i);

                    output[i] = PyFloat_AsDouble(pItem);
                }

            }

            Py_XDECREF(pArgs);
            Py_XDECREF(pResult);
        }
        Py_XDECREF(pMethod);
    } else {
        printf("hitpoint pointer was null\n");
    }
    // Unlock the mutex to allow other threads to access the model
    pthread_mutex_unlock(&model_mutex);
    
}

/*
THIS IS USED TO PASS IN A RAY ORIGIN AND AZ EL AND GET THE OUTPUT
REQUIRED FOR SHADING. 
*/

void NeuralRayTracer_GetShading(NeuralRayTracer* instance, double origin[3], double az_el[2], double output[3]) {
    
    // Lock the mutex to ensure exclusive access to the model
    pthread_mutex_lock(&model_mutex);

    if (instance->pModel2Instance != NULL) {

        int present = PyObject_HasAttrString(instance->pModel2Instance, "predict_dist_az_el");

        PyObject* pMethod = PyObject_GetAttrString(instance->pModel2Instance, "predict_dist_az_el");

        if (PyCallable_Check(pMethod)) {

            PyObject* pArgs = PyTuple_Pack(2, 
                                           Py_BuildValue("[d,d,d]", origin[0], origin[1], origin[2]),
                                           Py_BuildValue("[d,d]", az_el[0], az_el[1]));
            
            PyObject* pResult = PyObject_CallObject(pMethod, pArgs);

            if (PyList_Check(pResult) && PyList_Size(pResult) == 3) {

                for (int i = 0; i < 3; i++) {

                    PyObject* pItem = PyList_GetItem(pResult, i);

                    output[i] = PyFloat_AsDouble(pItem);
                }

            }

            Py_XDECREF(pArgs);
            Py_XDECREF(pResult);
        }
        Py_XDECREF(pMethod);
    } else {
        printf("shading pointer was null\n");
    }

    // Unlock the mutex to allow other threads to access the model
    pthread_mutex_unlock(&model_mutex);
    
}
