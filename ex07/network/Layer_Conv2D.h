/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"

template
<
  int InX, int InY, int InC, //input image: x:width, y:height, c:color channels
  int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:color channels
  int OpX, int OpY //output img: x:width, y:height, same color channels as KnC
>
struct Conv2DLayer: public Layer
{
  Params* allocate_params() const override {
    //number of kernel parameters:
    // 2d kernel size * number of inp channels * number of out channels
    const int nParams = KnY * KnX * InC * KnC;
    const int nBiases = KnC;
    return new Params(nParams, nBiases);
  }

  Conv2DLayer(const int _ID) : Layer(OpX * OpY * KnC, _ID) {
    static_assert(InX>0 && InY>0 && InC>0, "Invalid input");
    static_assert(KnX>0 && KnY>0 && KnC>0, "Invalid kernel");
    static_assert(OpX>0 && OpY>0, "Invalid outpus");
    print();
  }

  void print() {
    printf("(%d) Conv: In:[%d %d %d %d %d] F:[%d %d %d %d] Out:[%d %d %d]\n",
      ID, OpY,OpX,KnY,KnX,InC, KnY,KnX,InC,KnC, OpX,OpY,KnC);
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override
  {
    assert(act[ID]->layersSize   == OpY * OpX *                   KnC);
    assert(act[ID-1]->layersSize == OpY * OpX * KnY * KnX * InC      );
    assert(param[ID]->nWeights   ==             KnY * KnX * InC * KnC);
    assert(param[ID]->nBiases    ==                               KnC);

    const int batchSize = act[ID]->batchSize;
    const Real* const INP = act[ID-1]->output;
          Real* const OUT = act[ID]->output;

    // printf("TO CHECK: Conv2DLayer::forward\n");
    const Real* const weight= param[ID]->weights;
    const Real* const   bias= param[ID]->biases;

#pragma omp parallel for schedule(static)
    for (int b= 0; b < batchSize * OpY * OpX; b++)
      for (int n= 0; n < KnC; n++)
        OUT[b * KnC + n]= bias[n];

    gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
         batchSize * OpY * OpX, KnC, KnY * KnX * InC,
         (Real)1.0, INP, KnY * KnX * InC,
         weight, KnC,
         (Real)1.0, OUT, KnC);
  }
  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad) const override
  {
    const int batchSize = act[ID]->batchSize;
    const Real* const dEdO = act[ID]->dError_dOutput;

    // printf("TO CHECK: Conv2DLayer::bckward\n");
    const Real* const INP = act[ID-1]->output; //  
    const Real* const weight = param[ID]->weights; //


    // TO CHECK: Implement BackProp to compute bias gradient: dError / dBias
    {
      Real* const grad_B = grad[ID]->biases; // size KnC
      std::fill(grad_B, grad_B + KnC, 0);
#pragma omp parallel for schedule(static, 64/sizeof(Real))
      for (int n= 0; n < KnC; n++)
        for (int b= 0; b < batchSize * OpY * OpX; b++)
          grad_B[n] += dEdO[b * KnC + n];
    }

    // TO CHECK: Implement BackProp to compute weight gradient: dError / dWeights
    {
      Real* const grad_W = grad[ID]->weights; // KnY*KnX*InC * KnC 
      gemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          KnY * KnX * InC, KnC, batchSize * OpY * OpX,
          (Real)1.0, INP, KnY * KnX * InC,
                     dEdO, KnC,
          (Real)0.0, grad_W, KnC);
    }

    // TO CHECK: Implement BackProp to compute dEdO of previous layer
    {
      Real* const errinp = act[ID-1]->dError_dOutput;  
      gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
          batchSize * OpY * OpX, KnY * KnX * InC, KnC, 
          (Real)1.0, dEdO, KnC,
                     weight, KnC,
          (Real)0.0, errinp, KnY * KnX * InC);
    }
  }

  void init(std::mt19937& gen, const std::vector<Params*>& param) const override
  {
    // get pointers to layer's weights and bias
    Real *const W = param[ID]->weights, *const B = param[ID]->biases;
    // initialize weights with Xavier initialization
    const int nAdded = KnX * KnY * InC, nW = param[ID]->nWeights;
    const Real scale = std::sqrt(6.0 / (nAdded + KnC));
    std::uniform_real_distribution < Real > dis(-scale, scale);
    std::generate(W, W + nW, [&]() {return dis( gen );});
    std::fill(B, B + KnC, 0);
  }
};
