#ifndef NEURAL_RAY_TRACER_H
#define NEURAL_RAY_TRACER_H

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct NeuralRayTracer {
    PyObject *pName, *pModule, *pModelClass, *pModelClassReg, *pModelInstance, *pModel2Instance;
} NeuralRayTracer;

NeuralRayTracer* NeuralRayTracer_Create(const char* model_path);
void NeuralRayTracer_Destroy(NeuralRayTracer* instance);
void NeuralRayTracer_ShootRay(NeuralRayTracer* instance, double origin[3], double az_el[2], double output[1]);
void NeuralRayTracer_GetShading(NeuralRayTracer* instance, double origin[3], double az_el[2], double output[3]);
#endif // NEURAL_RAY_TRACER_H
