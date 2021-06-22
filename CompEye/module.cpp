#include <Python.h>
#include <arrayobject.h>
#include <iostream>

// Used as a guide: https://github.com/johnnylee/python-numpy-c-extension-examples/blob/master/src/simple2.c

static PyObject* antiHebUpdate(PyObject*, PyObject* args) {
	int inN, outN;
	float lr;
	PyArrayObject* weightsObject;
	PyArrayObject* hiddensObject;
	PyArrayObject* actsObject;
	npy_float32* w;
	npy_float32* h;
	npy_float32* y;
	npy_float32* preSum;

	if (!PyArg_ParseTuple(args, "llfO!O!O!",
			&inN,
			&outN,
			&lr,
			&PyArray_Type, &weightsObject,
			&PyArray_Type, &hiddensObject,
			&PyArray_Type, &actsObject)) { // Convert to PyArray_Type first
		return NULL;
	}

	w = (npy_float32*)PyArray_DATA(weightsObject);
	h = (npy_float32*)PyArray_DATA(hiddensObject);
	y = (npy_float32*)PyArray_DATA(actsObject);

	npy_float32* sumArr = (npy_float32*)malloc(4 * inN);

	float reg = lr / inN;

	for (int c = 0; c < inN; c++) {
		sumArr[c] = 0;
		for (int k = 0; k < outN; k++) {
			sumArr[c] += h[k] * w[(k * outN) + c];
		} 
	}

	int index = 0;
	// Could put this so that it does per column and calculate sum without array, but makes vectorization difficult
	for (int r = 0; r < outN; r++) {
		for (int c = 0; c < inN; c++) {
			//std::cout << (r * outN) + c;
			index = (r * outN) + c;
			w[index] = w[index] + reg * (w[index] - (y[r] + h[r]) * sumArr[c]);
		}
	}

	free(sumArr);

	Py_RETURN_NONE;
}

static PyMethodDef BeeLib_methods[] = {
	{ "antiHebUpdate", (PyCFunction)antiHebUpdate, METH_VARARGS, nullptr },

	{ NULL, NULL, 0, NULL }
};

static PyModuleDef BeeLib_module = {
	PyModuleDef_HEAD_INIT,
	"BeeLib",
	"A python library for biological computation, especially navigation.",
	0,
	BeeLib_methods
};

PyMODINIT_FUNC PyInit_BeeLib() {
	import_array();
	return PyModule_Create(&BeeLib_module);
}