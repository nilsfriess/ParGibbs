#include "parmgmc/iact.h"

#include <petscsys.h>

#include <fftw3.h>

PetscErrorCode Autocorrelation(PetscInt n, const PetscScalar *x, PetscScalar **acf)
{
  fftw_plan     p;
  fftw_complex *in, *out;
  PetscScalar   mean = 0;

  PetscFunctionBeginUser;
  in  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n);
  out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n);

  for (PetscInt i = 0; i < n; ++i) mean += 1. / n * x[i];
  for (PetscInt i = 0; i < n; ++i) in[i] = x[i] - mean;

  p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  for (PetscInt i = 0; i < n; ++i) out[i] = out[i] * conj(out[i]);
  p = fftw_plan_dft_1d(n, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  PetscCall(PetscMalloc1(n, acf));
  for (PetscInt i = 0; i < n; ++i) (*acf)[i] = PetscRealPart(in[i]) / PetscRealPart(in[0]);

  fftw_free(in);
  fftw_free(out);
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

PetscErrorCode IACT(PetscInt n, const PetscScalar *x, PetscScalar *tau)
{
  PetscScalar *out;
  PetscInt     w;

  PetscFunctionBeginUser;
  PetscCall(Autocorrelation(n, x, &out));
  for (PetscInt i = 1; i < n; ++i) out[i] = out[i] + out[i - 1];
  for (PetscInt i = 1; i < n; ++i) out[i] = 2 * out[i] - 1;
  PetscCall(AutoWindow(n, out, 5, &w));
  *tau = out[w];
  PetscCall(PetscFree(out));
  PetscFunctionReturn(PETSC_SUCCESS);
}
