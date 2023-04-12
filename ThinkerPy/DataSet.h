#pragma once
class DataSet
{
public:
	float **inputs;
	float **outputs;
	DataSet(float**inputs, float** outputs) {
		this->inputs = inputs;
		this->outputs = outputs;
	}
	~DataSet() {
		this->inputs = nullptr;
		this->outputs = nullptr;
		delete this->inputs;
		delete this->outputs;
	}
};