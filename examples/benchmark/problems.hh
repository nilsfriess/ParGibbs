#pragma once

#include <petscsystypes.h>
#include <petscmat.h>
#include <petscvec.h>

class Problem {
public:
  virtual PetscErrorCode GetPrecisionMat(Mat *)   = 0;
  virtual PetscErrorCode GetRHSVec(Vec *)         = 0;
  virtual PetscErrorCode GetMeasurementVec(Vec *) = 0;
  virtual PetscErrorCode VisualiseResults(Vec mean = nullptr, Vec var = nullptr) = 0;

  virtual ~Problem() = default;
};
