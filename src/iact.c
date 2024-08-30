#include "parmgmc/iact.h"

#include <complex.h>
#include <fftw3.h>

#include <petscsys.h>
#include <petscerror.h>
#include <petscmath.h>

static PetscInt NextPowTwo(PetscInt n)
{
  PetscInt i = 1;
  while (i < n) i <<= 1;
  return i;
}

PetscErrorCode Autocorrelation(PetscInt n, const PetscScalar *x, PetscScalar **acf)
{
  fftw_plan     p;
  fftw_complex *in, *out;
  PetscScalar   mean = 0;
  PetscInt      N    = NextPowTwo(n);

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(2 * N, &in));
  PetscCall(PetscCalloc1(2 * N, &out));

  for (PetscInt i = 0; i < n; ++i) mean += 1. / n * x[i];
  for (PetscInt i = 0; i < n; ++i) in[i] = x[i] - mean;

  p = fftw_plan_dft_1d(2 * N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  for (PetscInt i = 0; i < 2 * N; ++i) out[i] = out[i] * conj(out[i]);
  p = fftw_plan_dft_1d(2 * N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  PetscCall(PetscMalloc1(n, acf));
  for (PetscInt i = 0; i < n; ++i) (*acf)[i] = PetscRealPart(in[i]) / PetscRealPart(in[0]);

  PetscCall(PetscFree(in));
  PetscCall(PetscFree(out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AutoWindow(PetscInt n, const PetscScalar *taus, PetscInt c, PetscInt *w)
{
  PetscBool flag = PETSC_FALSE;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < n; ++i) {
    if (i < c * taus[i]) {
      flag = PETSC_TRUE;
      break;
    }
  }
  if (flag) {
    flag = PETSC_FALSE;
    for (PetscInt i = 0; i < n; ++i) {
      if (i >= c * taus[i]) {
        *w   = i;
        flag = PETSC_TRUE;
        break;
      }
    }
    if (flag == PETSC_FALSE) *w = 0;
  } else *w = n - 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IACT(PetscInt n, const PetscScalar *x, PetscScalar *tau, PetscScalar **acf, PetscBool *valid)
{
  PetscScalar *out;
  PetscInt     w;

  PetscFunctionBeginUser;
  PetscCheck(n > 1, MPI_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Too few data points");
  PetscCall(Autocorrelation(n, x, &out));
  if (acf) {
    PetscCall(PetscMalloc1(n, acf));
    PetscCall(PetscArraycpy(*acf, out, n));
  }
  for (PetscInt i = 1; i < n; ++i) out[i] = out[i] + out[i - 1];
  for (PetscInt i = 0; i < n; ++i) out[i] = 2 * out[i] - 1;
  PetscCall(AutoWindow(n, out, 5, &w));
  *tau = out[w];
  if (valid) *valid = 500 * (*tau) <= n;
  PetscCall(PetscFree(out));
  PetscFunctionReturn(PETSC_SUCCESS);
}
