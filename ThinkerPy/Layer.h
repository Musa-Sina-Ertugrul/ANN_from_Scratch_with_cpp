#pragma once
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <cmath>
#include "Enums.h"
class Layer
{
public:
	static int layerNumber;
	float **inputs;
	float **outputsActiveted;
	float** outputs;
	float **weights;
	LayerF* fType;
	float gradient[];
	Layer(int in, int out,LayerF* fType) {
		this->outputs = new float*[out];
		this->outputsActiveted = new float* [out];
		this->weights = new float* [out];
		for (int i = 0; i < out; i++) {
			this->weights[i] = new float[in];
			for (int j = 0; j < in; j++) {
				this->weights[i][j] = 0.000000;
			}
		}
		this->gradient[out];
		this->initWeights(in, out);
		this->fType = fType;
		printf("done { %d }",this->layerNumber);
		this->layerNumber++;
	}
	~Layer() {
		delete[] this->inputs;
		delete[] this->outputs;
		for (int i = 0; i < sizeof(this->weights); i++) {
			delete[] this->weights[i];
		}
		delete[] this->weights;
		this->inputs = nullptr;
		this->outputs = nullptr;
		this->weights = nullptr;
		
	}

private:
	void initWeights(int& in, int& out) {
		gsl_rng* r = gsl_rng_alloc(gsl_rng_default);
		float std = sqrt(2.000000 / ((float)(in + out)));
		int len = sizeof(this->weights[0]);
		for (int i = 0; i < sizeof(this->weights); i++) {
			for (int j = 0; j < len; j++) {
				this->weights[i][j] = float(gsl_ran_gaussian(r, std));
			}
		}
		delete r;
		r = nullptr;
	}
};

int Layer::layerNumber = int(0);

