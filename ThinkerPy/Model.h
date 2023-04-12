#pragma once
#include "DataSet.h"
#include "Layer.h"
#include "Enums.h"
#include <string>
#include <gsl/gsl_math.h>
#include <math.h>
#include <stdio.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_blas.h>
#include <random>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class Model
{
public:
	DataSet* datas;
	int epochs;
	bool bouncingLR;
	float learningRate;
	int batchSize;
	Layer** layers;
	LossFType* lossType;
	bool batchNorm;
	float dropOutRate;
	RegFType* regType;
	float regLambda;
	explicit Model(DataSet* datas, int&& epochs, bool&& bouncingLR, float&& learningRate,
		int&& batchSize, Layer** layers, LossFType&& lossType, bool&& batchNorm, float&& dropOutRate,
		RegFType&& regType, float&& regLambda) {

		this->datas = datas;
		this->epochs = epochs;
		this->bouncingLR = bouncingLR;
		this->learningRate = learningRate;
		this->batchSize = batchSize;
		this->layers = layers;
		this->lossType = &lossType;
		this->batchNorm = batchNorm;
		this->dropOutRate = dropOutRate;
		this->regType = &regType;
		this->regLambda = regLambda;
		this->error = float(0.000000);
		this->currentEpoch = 0;
		this->currentLR = this->learningRate;
		int percentage = static_cast<int>(10 * this->dropOutRate);
		int count(0);
		for (int i = 0; i < 10; i++) {
			this->dropOutArray[i] = 1;
		}
		for (int i = 0; i < 10; i++) {
			if (count == percentage) {
				break;
			}
			int tmp = (*Model::dist01)(*Model::mt);
			if (tmp == 0) {
				count++;
				this->dropOutArray[i] = 0;
			}
		}
		this->currentLayer = 0;
		this->pastMomentum = 0.000000;
		this->pastVelocity = 0.000000;
	}
	~Model() {
		this->datas = nullptr;
		delete[] this->layers;
		this->layers = nullptr;
		delete this->lossType;
		this->lossType = nullptr;
		delete this->regType;
		this->regType = nullptr;
	}

	float** getOutputs() {
		return this->layers[sizeof(this->layers) - 1]->outputsActiveted;
	}
	void trainModel() {
		int trues = 0;
		int currentBatch = 0;
		for (int epoch = 0; epoch < this->epochs; epoch++) {
			this->currentEpoch = epoch % sizeof(this->datas->inputs);
			trues += this->forward(this->datas->inputs[this->currentEpoch]);
			if (epoch + 1 == this->batchSize) {
				this->error /= this->batchSize;
				currentBatch++;
				cout << "error :" << this->error << " current batch : " << currentBatch << " correctness : " << ((float)trues) / ((float)this->batchSize) << endl;
				trues = 0;
				
			}
			this->backward();
		}
	}
	float** runModel(float** inputs);

private:
	static random_device rd;
	static mt19937* mt;
	static uniform_int_distribution<int>* dist01;
	static uniform_int_distribution<int>* distIndex;
	int dropOutArray[10];
	int currentEpoch;
	int currentLayer;
	float currentLR;
	float error;
	float beta1 = float(0.9);
	float beta2 = float(0.999);
	const float epsilon = float(0.000001);
	float pastVelocity;
	float pastMomentum;

	void adam(int& i, int& j, int& k, float* hashMap[]) {
		this->layers[this->currentLayer]->weights[i][j] = this->layers[this->currentLayer]->weights[i][j] - (this->momentumHat(i,j,k, hashMap) * (this->currentLR / (sqrt(this->velocityHat(i,j,k, hashMap)) + this->epsilon)));
	}

	float momentum(int& i, int& j, int& k, float* hashMap[]) {
		if (this->currentEpoch == 0) {
			this->pastMomentum = this->gradient(i,j,k, hashMap);
			return this->pastMomentum;
		}
		this->pastMomentum = this->beta1 * this->pastMomentum + (1 - this->beta1) * this->gradient(i,j,k, hashMap);
		return this->pastMomentum;
	}
	float gradient(int& i, int& j, int& k, float* hashMap[]) {
		float tmp;
		if (this->currentLayer == sizeof(this->layers) - 1) {
			if (hashMap[i] != nullptr) {
				tmp = *hashMap[i];
				return tmp * this->layers[this->currentLayer]->weights[i][j];
			}
			tmp = this->dLoss(this->datas->outputs[this->currentEpoch][i], *this->layers[this->currentLayer]->outputsActiveted[i]) * this->dActivation(*this->layers[this->currentLayer]->outputs[i]);
			this->layers[this->currentLayer]->gradient[k] += tmp;
			hashMap[i] = new float(tmp);
			return tmp * this->layers[this->currentLayer]->weights[i][j];
		}
		if (hashMap[i] != nullptr) {
			tmp = *hashMap[i];
			return tmp * this->layers[this->currentLayer]->weights[i][j];
		}
		tmp = this->layers[this->currentLayer + 1]->gradient[i] * this->dActivation(*this->layers[this->currentLayer]->outputs[i]);
		this->layers[this->currentLayer]->gradient[k] += tmp;
		hashMap[i] = new float(tmp);
		return tmp * this->layers[this->currentLayer]->weights[i][j];
	}
	float velocity(int& i, int& j, int& k, float* hashMap[]) {
		if (this->currentEpoch == 0) {
			this->pastVelocity = this->gradient(i,j,k, hashMap) * this->gradient(i,j,k, hashMap);
			return this->pastVelocity;
		}
		this->pastVelocity = this->beta2 * this->pastVelocity + (1 - this->beta2) * (this->gradient(i,j,k, hashMap)*this->gradient(i,j,k, hashMap));
		return this->pastVelocity;

	}
	float momentumHat(int& i, int& j, int& k,float* hashMap[]) {
		return this->momentum(i,j,k, hashMap) / 1 - pow(this->beta1, this->currentEpoch);
	}
	float velocityHat(int& i, int& j, int& k,float* hashMap[]) {
		return this->velocity(i,j,k, hashMap) / 1 - pow(this->beta2, this->currentEpoch);
	}
	float& tanh(float& x) {
		x = (gsl_sf_exp(x) - gsl_sf_exp(-x))/ (gsl_sf_exp(x)+gsl_sf_exp(x));
		return x;
	}
	float& cosBounce() {
		this->currentLR = gsl_sf_cos(this->learningRate * this->currentEpoch);
	}
	float& sigmoid(float& x) {
		x = 1 / (1 + gsl_sf_exp(-x));
		return x;
	}
	int forward(float inputs[]) {
		float** tmpInputs;
		for (int i = 0; i < sizeof(this->layers); i++) {
			this->currentLayer = i;
			if (i == 0) {
				this->feedLayer(inputs);
			}
			else {
				this->feedLayer(tmpInputs);
			}
			for (int j = 0; j<sizeof(*this->layers[i]->gradient); j++) {
				this->layers[i]->gradient[j] = 0.000000;
			}
			for (int j = 0; j < sizeof(this->layers[i]->outputs); j++) {
				*this->layers[i]->outputs[j] = 0.000000;
				for (int k = 0; k < sizeof(this->layers[i]->weights[j]); k++) {
					*this->layers[i]->outputs[j] += this->layers[i]->weights[j][k] * *this->layers[i]->inputs[k];
				}
				this->layers[i]->outputsActiveted[j] = new float(this->activation(*this->layers[i]->outputs[j]));
				if (this->currentLayer == sizeof(this->layers) - 1) {
					this->error += this->loss(this->datas->outputs[this->currentEpoch][j], *this->layers[i]->outputs[j]);
				}
			}
			tmpInputs = this->layers[i]->outputs;
		}
		return this->checkTrues();
	}
	void backward() {
		for (int i = sizeof(this->layers) - 1; i > -1; i--) {
			this->currentLayer = i;
			if (this->bouncingLR) {
				this->currentLR = this->cosBounce();
			}
			for (int grad = 0; grad < sizeof(*this->layers[i]->gradient); grad++) {
				float* gradients[sizeof(this->layers[this->currentLayer]->weights)];
				for (int j = 0; j < sizeof(this->layers[this->currentLayer]->weights); j++) {
					bool addable = true;
					for (int k = 0; k < sizeof(this->layers[this->currentLayer]->weights[0]); k++) {
						this->adam(j, k, grad, gradients);
						addable = false;
						
					}
				}
				for (int j = 0; j < sizeof(this->layers[this->currentLayer]->weights); j++) {
					delete gradients[j];
					gradients[j] = nullptr;
				}
			}
		}
	}
	float& dTanh(float& x) {
		x = gsl_sf_exp(2 * x) / (float)(gsl_sf_exp(2 * x) + 1) * (gsl_sf_exp(2 * x)+1);
		return x;
	}
	float&& dRelu(float& x) {
		if (x > 0.0) {
			return 1.0;
		}
		return 0.0;
	}
	float& dSigmoid(float& x) {
		x = this->sigmoid(x) * (1 - this->sigmoid(x));
		return x;
	}
	Layer& feedLayer(float inputs[]) {
		int len = sizeof(inputs) + 1;
		float newInputs[sizeof(inputs) + 1];
		newInputs[0] = float(1.000000);
		for (int i = 1; i < len; i++) {
			newInputs[i] = inputs[i - 1];
		}
		float* tmp = newInputs;
		this->layers[this->currentLayer]->inputs = &tmp;
		return *this->layers[this->currentLayer];
	}
	Layer& feedLayer(float** inputs) {
		int len = sizeof(inputs) + 1;
		float newInputs[sizeof(inputs) + 1];
		newInputs[0] = float(1.000000);
		for (int i = 1; i < len; i++) {
			newInputs[i] = *inputs[i - 1];
		}
		float *tmp = newInputs;
		this->layers[this->currentLayer]->inputs = &tmp; 
		return *this->layers[this->currentLayer];
	}
	float& relu(float& x) {
		if (x > 0.0) {
			return x;
		}
		x = 0.0;
		return x;
	}
	float activation(float& y) {
		switch (*this->layers[this->currentLayer]->fType)
		{
		case Tanh:
			return this->tanh(y);
			break;
		case Sigmoid:
			return this->sigmoid(y);
			break;
		case Relu:
			return this->relu(y);
			break;
		default:
			throw "type correct activation func";
			break;
		}
	}
	float dActivation(float& y) {
		switch (*this->layers[this->currentLayer]->fType)
		{
		case Tanh:
			return this->dTanh(y);
			break;
		case Sigmoid:
			return this->dSigmoid(y);
			break;
		case Relu:
			return this->dRelu(y);
			break;
		default:
			throw "type correct activation func";
			break;
		}
	}
	float bCELoss(float& yHat, float& y) {
		return -(yHat * gsl_sf_log(y) + (1 - yHat * (1 - gsl_sf_log(1 - y))));
	}
	float cELoss(float& yHat, float& y) {
		return yHat * gsl_sf_log(y);
	}
	float l1Loss(float& yHat, float& y) {
		return abs(yHat - y);
	}
	float l2Loss(float& yHat, float& y) {
		return (yHat - y)* (yHat - y);
	}
	float loss(float& yHat, float& input) {
		switch (*this->lossType)
		{
		case BCELoss:
			return this->bCELoss(yHat,input);
			break;
		case CELoss:
			return this->cELoss(yHat,input);
			break;
		case L1Loss:
			return this->l1Loss(yHat,input);
			break;
		case L2Loss:
			return this->l2Loss(yHat,input);
			break;
		default:
			throw "type correct loss func";
			break;
		}
	}
	float dLoss(float& yHat,float& y) {
		switch (*this->lossType)
		{
		case BCELoss:
			return this->dBCELoss(y,yHat);
			break;
		case CELoss:
			return this->dCELoss(y, yHat);
			break;
		case L1Loss:
			return this->dl1Loss(y, yHat);
			break;
		case L2Loss:
			return this->dl2Loss(y, yHat);
			break;
		default:
			throw "type correct loss func";
			break;
		}
	}
	float dBCELoss(float& yHat,float& y) {
		return -(yHat / y - (1 - yHat) / (1 - y));
	}
	float dCELoss(float& yHat,float& y) {
		return -((yHat / y) - ((1 - yHat) / (1 - y)));
	}
	float dl1Loss(float& yHat,float& y) {
		if (y > yHat) {
			return 1.0;
		}
		if (yHat > y) {
			return -1.0;
		}
		return 0.0;
	}
	float dl2Loss(float& yHat,float& y) {
		return -(yHat - y);
	}
	void bachNorm() {
		float total = 0.000000;
		int len = sizeof(this->layers[this->currentLayer]->outputs);
		for (int i = 0; i < len;i++ ) {
			total += *this->layers[this->currentLayer]->outputs[i];
		}
		total /= len;
		float meansqr = 0.000000;
		for (int i = 0; i < len; i++) {
			meansqr += (*this->layers[this->currentLayer]->outputs[i] - total)* (*this->layers[this->currentLayer]->outputs[i] - total);
		}
		float std = sqrt(meansqr / (len - 1));
		for (int i = 0; i < len; i++) {
			*this->layers[this->currentLayer]->outputs[i] = (*this->layers[this->currentLayer]->outputs[i]-total)/std;
		}

	}
	void dropOut() {
		int len = sizeof(this->layers[this->currentLayer]->inputs);
		for (int i = 0; i < len; i++) {
			int dropping = (*Model::distIndex)(*Model::mt);
			*(this->layers[this->currentLayer]->inputs[i]) *= dropping;
		}
	}
	float l1Reg(float** weights) {
		float total(0.000000);
		int row(sizeof(weights));
		int col(sizeof(weights[0]));
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				total += abs(weights[i][j]);
			}
		}
		return this->regLambda*total;
	}
	float l2Reg(float** weights) {
		float total(0.000000);
		int row(sizeof(weights));
		int col(sizeof(weights[0]));
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				total += weights[i][j]*weights[i][j];
			}
		}
		return this->regLambda * total;
	}
	float reg(float** weights) {
		switch (*this->regType)
		{
		case None:
			return 0.0;
			break;
		case L1:
			return this->l1Reg(weights);
			break;
		case L2:
			return this->l2Reg(weights);
			break;
		default:
			throw "type correct reg type";
			break;
		}
	}
	int checkTrues() {
		switch (*this->lossType)
		{
		case BCELoss:
			return this->checkTruesBCE();
			break;
		case CELoss:
			return this->checkTruesOthers();
			break;
		default:
			throw "type correct loss type";
			break;
		}
	}
	int checkTruesBCE() {
		int answer=0;
		if (*this->layers[sizeof(this->layers) - 1]->outputsActiveted[0] > 0.5) {
			answer = 1;
		}
		if (answer == *this->datas->outputs[this->currentEpoch]) {
			return 1;
		}
		return 0;
	}
	int checkTruesOthers() {
		float max = this->layers[sizeof(this->layers) - 1]->outputs[this->currentEpoch][0];
		int index = 0;
		for (int i = 1; i < sizeof(this->layers[sizeof(this->layers) - 1]->outputs[this->currentEpoch]); i++) {
			if (max < this->layers[sizeof(this->layers) - 1]->outputs[this->currentEpoch][i]) {
				index = i;
				max = this->layers[sizeof(this->layers) - 1]->outputs[this->currentEpoch][i];
			}
		}
		if (this->datas->outputs[this->currentEpoch][index] == 1) {
			return 1;
		}
		return 0;
	}

};

mt19937* Model::mt = new mt19937(Model::rd());
uniform_int_distribution<int>* Model::dist01 = new uniform_int_distribution<int>(0,1);
uniform_int_distribution<int>* Model::distIndex = new uniform_int_distribution<int>(-1, 2);