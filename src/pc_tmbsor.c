#include "parmgmc/pc/pc_tmbsor.h"
#include "mpi_proto.h"

#include <assert.h>
#include <mpi.h>

#include <petsc/private/pcimpl.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

typedef struct {
  PetscBool done;
  PetscInt  ndep;
} *MidNode;

typedef struct {
  PetscInt   idx;
  VecScatter sct;
  Vec        v;
} *BotWithMidNb;

typedef struct {
  VecScatter sct;
  Vec        vec;
  IS         ids;
} *NodeSet;

typedef struct {
  PetscInt *proccolmap;

  NodeSet top, bot_nomid;

  IS int1, int2;

  MidNode *mid,       // mid nodes that have both lower and higher mid nodes as neighbours
    *mid_no_high_mid, // mid nodes that have no higher mid nodes as neighbours
    *mid_no_low_mid;  // mid nodes that have no lower mid nodes as neighbours

  PetscInt nmid, nmid_no_high_mid, nmid_no_low_mid;
} *TMBSOR;

static PetscErrorCode PCDestroy_TMBSOR(PC pc)
{
  TMBSOR tmb = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PetscFree(tmb->proccolmap));

  PetscCall(VecScatterDestroy(&tmb->top->sct));
  PetscCall(VecDestroy(&tmb->top->vec));
  PetscCall(ISDestroy(&tmb->top->ids));
  PetscCall(PetscFree(tmb->top));

  /* PetscCall(VecScatterDestroy(&tmb->bot_nomid->sct)); */
  /* PetscCall(VecDestroy(&tmb->bot_nomid->vec)); */
  /* PetscCall(ISDestroy(&tmb->bot_nomid->ids)); */
  /* PetscCall(PetscFree(tmb->bot_nomid)); */

  PetscCall(ISDestroy(&tmb->int1));
  PetscCall(ISDestroy(&tmb->int2));

  PetscCall(PetscFree(tmb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TMBSOR_ColorProcessors(PC pc)
{
  TMBSOR          tmb = pc->data;
  PetscMPIInt     size, rank;
  Mat             Ap, Ao, P;
  const PetscInt *colmap, *ii, *jj;
  PetscInt        n, N, *procmap, n_nb_total = 0, *proc_cols;
  PetscScalar    *proc_cols_vals;
  PetscLayout     layout;
  MatColoring     proc_coloring;
  ISColoring      pcol;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));
  PetscCall(PetscCalloc1(size, &procmap));

  PetscCall(MatMPIAIJGetSeqAIJ(pc->pmat, &Ap, &Ao, &colmap));
  PetscCall(MatGetSize(Ao, &n, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(Ao, &ii, &jj, NULL, NULL));
  PetscCall(MatGetLayouts(pc->pmat, NULL, &layout));
  PetscCall(MatGetSize(pc->pmat, &N, NULL));

  /* Create map (of size `size` = no. of MPI ranks) that stores if a certain index
     (= MPI rank) has values that our process needs during Gauss-Seidel sweeps.
  */
  for (PetscInt i = 0; i < n; ++i) {
    for (PetscInt j = ii[i]; j < ii[i + 1]; ++j) {
      PetscInt    c = colmap[jj[j]];
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(layout, c, &owner));
      procmap[owner] = 1;
    }
  }

  /* Find out how many MPI neighbours we have in total. */
  for (PetscInt i = 0; i < size; ++i)
    if (procmap[i] > 0) n_nb_total++;

  /* Say the map above looks like this for rank 0:
        procmap = {0, 1, 0, 1}.
     This means that rank 1 and rank 4 are neighbours. Below we transform this
     into the form
        proc_cols = {1, 4}.
  */
  PetscCall(PetscCalloc1(n_nb_total, &proc_cols));
  PetscCall(PetscCalloc1(n_nb_total, &proc_cols_vals));
  for (PetscInt i = 0, j = 0; i < size; ++i) {
    if (procmap[i] > 0) {
      proc_cols[j]      = i;
      proc_cols_vals[j] = 1;
      ++j;
    }
  }

  /* Finally we create a matrix whose matrix graphs represents the "MPI
     neighbouring layout" that we determined above. This matrix is always
     of size `#MPI ranks x #MPI ranks` and each process owns exactly one row.
     The columns contain a non-zero value (the value is irrelevant, we use 1)
     if the MPI rank corresponding to the column is a MPI neighbour of
     the current MPI rank. We do all this so that we can use PETSc's
     MatColoring to create a colouring of the processors.
   */
  PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)pc), 1, 1, size, size, 0, NULL, n_nb_total, NULL, &P));
  for (PetscInt i = 0; i < size; ++i) {
    const PetscInt row = rank;
    PetscCall(MatSetValues(P, 1, &row, n_nb_total, proc_cols, proc_cols_vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(P, NULL, "-mat_procs_view"));

  PetscCall(MatColoringCreate(P, &proc_coloring));
  PetscCall(MatColoringSetDistance(proc_coloring, 1));
  PetscCall(MatColoringSetType(proc_coloring, MATCOLORINGJP));
  PetscCall(MatColoringApply(proc_coloring, &pcol));
  PetscCall(ISColoringSetType(pcol, IS_COLORING_GLOBAL));
  PetscCall(ISColoringViewFromOptions(pcol, NULL, "-proc_coloring_view"));

  /* Lastly, create a map (in the form of an array of length `size`) that maps
     MPI ranks to colour indices. */
  PetscCall(PetscMalloc1(size, &tmb->proccolmap));
  for (PetscInt i = 0; i < size; ++i) {
    IS      *iss;
    PetscInt ncols;

    PetscCall(ISColoringGetIS(pcol, PETSC_USE_POINTER, &ncols, &iss));
    for (PetscInt c = 0; c < ncols; ++c) {
      const PetscInt *idxs;
      IS              gis;

      PetscCall(ISAllGather(iss[c], &gis));
      PetscCall(ISGetSize(gis, &n));
      PetscCall(ISGetIndices(gis, &idxs));
      for (PetscInt j = 0; j < n; ++j) {
        if (idxs[j] == i) tmb->proccolmap[i] = c; // Process i has color c
      }
      PetscCall(ISRestoreIndices(gis, &idxs));
      PetscCall(ISDestroy(&gis));
    }
    PetscCall(ISColoringRestoreIS(pcol, PETSC_USE_POINTER, &iss));
  }

  PetscCall(MatColoringDestroy(&proc_coloring));
  PetscCall(ISColoringDestroy(&pcol));
  PetscCall(PetscFree(proc_cols));
  PetscCall(PetscFree(proc_cols_vals));
  PetscCall(PetscFree(procmap));
  PetscCall(MatDestroy(&P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TMBSOR_CreateScatterForIndices(IS is, Mat A, VecScatter *sct, Vec *v)
{
  PetscInt        noff = 0, nidxs, *offidxs, cnt = 0;
  const PetscInt *idxs, *colmap, *ii, *jj;
  Mat             Ao;
  IS              sctis;
  Vec             gvec;

  PetscFunctionBeginUser;
  PetscCall(MatMPIAIJGetSeqAIJ(A, NULL, &Ao, &colmap));
  PetscCall(MatSeqAIJGetCSRAndMemType(Ao, &ii, &jj, NULL, NULL));

  // First, count number of neighbouring values on remote ranks
  PetscCall(ISGetLocalSize(is, &nidxs));
  PetscCall(ISGetIndices(is, &idxs));
  for (PetscInt i = 0; i < nidxs; ++i) noff += ii[idxs[i] + 1] - ii[idxs[i]];

  // Now we can get the actual indices of the neighbouring values
  PetscCall(PetscMalloc1(noff, &offidxs));
  for (PetscInt i = 0; i < nidxs; ++i) {
    for (PetscInt j = ii[idxs[i]]; j < ii[idxs[i] + 1]; ++j) { offidxs[cnt++] = colmap[jj[j]]; }
  }
  PetscCall(ISRestoreIndices(is, &idxs));

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, noff, offidxs, PETSC_OWN_POINTER, &sctis));
  PetscCall(VecCreateSeq(MPI_COMM_SELF, noff, v));
  PetscCall(MatCreateVecs(A, &gvec, NULL));
  PetscCall(VecScatterCreate(gvec, sctis, *v, NULL, sct));
  PetscCall(VecDestroy(&gvec));
  PetscCall(ISDestroy(&sctis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TMBSOR_PartitionNodes(PC pc)
{
  TMBSOR          tmb = pc->data;
  PetscLayout     layout;
  PetscMPIInt     rank;
  Mat             Ad, Ao;
  const PetscInt *colmap, *ii, *jj;
  PetscInt        gstart, start, end, ntop = 0, nbot = 0, nmid = 0, nint = 0, intcnt = 0, topcnt = 0, botcnt = 0, midcnt = 0;
  PetscInt       *nodes, *topnodes, *botnodes, *midnodes, *intnodes, mpinbcnt = 0;
  PetscMPIInt     mpinb[50];
  enum NodeType {
    TOP,
    MID,
    BOT,
    INT
  };

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));
  PetscCall(MatGetLayouts(pc->pmat, NULL, &layout));
  PetscCall(MatMPIAIJGetSeqAIJ(pc->pmat, &Ad, &Ao, &colmap));
  PetscCall(MatSeqAIJGetCSRAndMemType(Ao, &ii, &jj, NULL, NULL));
  PetscCall(MatGetOwnershipRange(Ao, &start, &end));
  PetscCall(PetscCalloc1(end - start, &nodes));
  for (PetscInt i = 0; i < 50; ++i) mpinb[i] = -1;
  for (PetscInt i = start; i < end; ++i) {
    PetscBool istop = PETSC_FALSE, isbot = PETSC_FALSE; // if both are true, then this is a mid node,
                                                        // if neither is true, it's an interior node

    for (PetscInt j = ii[i]; j < ii[i + 1]; ++j) {
      PetscInt    c = colmap[jj[j]];
      PetscMPIInt owner;

      PetscCall(PetscLayoutFindOwner(layout, c, &owner));
      mpinb[mpinbcnt++] = owner;
      if (tmb->proccolmap[owner] < tmb->proccolmap[rank]) istop = PETSC_TRUE;
      if (tmb->proccolmap[owner] > tmb->proccolmap[rank]) isbot = PETSC_TRUE;
    }

    if (!istop && !isbot) {
      nint++;
      nodes[i] = INT;
    } else if (!istop && isbot) {
      nbot++;
      nodes[i] = BOT;
    } else if (istop && !isbot) {
      ntop++;
      nodes[i] = TOP;
    } else {
      nmid++;
      nodes[i] = MID;
    }
  }
  assert(ntop + nbot + nmid + nint == end - start && "Sum of number of partitioned nodes does not match total nodes");
  PetscCall(PetscIntView(end - start, nodes, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscMalloc1(ntop, &topnodes));
  PetscCall(PetscMalloc1(nbot, &botnodes));
  PetscCall(PetscMalloc1(nmid, &midnodes));
  PetscCall(PetscMalloc1(nint, &intnodes));

  for (PetscInt i = start; i < end; ++i) {
    switch (nodes[i]) {
    case TOP:
      topnodes[topcnt++] = i;
      break;
    case BOT:
      botnodes[botcnt++] = i;
      break;
    case MID:
      midnodes[midcnt++] = i;
      break;
    case INT:
      intnodes[intcnt++] = i;
      break;
    }
  }
  PetscCall(PetscFree(nodes));

  {
    /* Split interior nodes into two parts such that approximately
            cost(int1) + cost(top) = cost(int2) + cost(bot)
       which is the same as
            cost(int1) = (cost(int) + cost(bot) - cost(top)) / 2,
       where cost(int) is the total cost of all interior nodes.
       Thus, we compute cost(int), cost(bot), and cost(top) and then decide
       where to split based on the second formula above. */
    PetscInt intcost = 0, topcost = 0, botcost = 0, tgt_int1_cost, curr_int1_cost = 0, splitidx;
    for (PetscInt i = 0; i < ntop; ++i) topcost += ii[topnodes[i] + 1] - ii[topnodes[i]];
    for (PetscInt i = 0; i < nbot; ++i) botcost += ii[botnodes[i] + 1] - ii[botnodes[i]];
    for (PetscInt i = 0; i < nint; ++i) intcost += ii[intnodes[i] + 1] - ii[intnodes[i]];

    tgt_int1_cost = roundf(0.5f * (intcost + botcost - topcost));
    for (splitidx = 0; splitidx < nint; ++splitidx) {
      curr_int1_cost += ii[intnodes[splitidx] + 1] - ii[intnodes[splitidx]];
      if (curr_int1_cost > tgt_int1_cost) break;
    }

    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc), splitidx, intnodes, PETSC_COPY_VALUES, &tmb->int1));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc), nint - splitidx, intnodes + splitidx + 1, PETSC_COPY_VALUES, &tmb->int2));
  }

  /* Create IS, VecScatter and ghost vector for TOP nodes. */
  PetscCall(PetscNew(&tmb->top));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc), ntop, topnodes, PETSC_OWN_POINTER, &tmb->top->ids));
  PetscCall(TMBSOR_CreateScatterForIndices(tmb->top->ids, pc->pmat, &tmb->top->sct, &tmb->top->vec));

  {
    /* Find out if there are BOT nodes that have MID nodes as neighbours.
       We do this by first sending the indices of all of our MID nodes to
       all of our neighbours (and receive the same info from all neighbours).
       Then we just check if one of our BOT nodes neighbours one of the
       received nodes.
    */
    MPI_Request *requests;
    PetscInt    *gmidnodes; // midnodes in global numbering
    PetscInt    *bot_midnb; // 1 for bot nodes that have mid nodes as neighbours
    PetscInt     nbot_midnb = 0;

    PetscCall(PetscCalloc1(nbot, &bot_midnb));

    PetscCall(PetscMalloc1(nmid, &gmidnodes));
    PetscCall(MatGetOwnershipRange(pc->pmat, &gstart, NULL));
    for (PetscInt i = 0; i < nmid; ++i) gmidnodes[i] = midnodes[i] + gstart;

    PetscCall(PetscMalloc1(mpinbcnt, &requests));
    for (PetscInt i = 0; i < mpinbcnt; ++i) PetscCallMPI(MPI_Isend(gmidnodes, nmid, MPIU_INT, mpinb[i], 0, PetscObjectComm((PetscObject)pc), &requests[i]));
    for (PetscInt i = 0; i < mpinbcnt; ++i) PetscCallMPI(MPI_Wait(&requests[i], MPI_STATUS_IGNORE));

    for (PetscInt i = 0; i < mpinbcnt; ++i) {
      // We don't know in which order the messages from our neighbours arrive, so we probe for them first
      MPI_Status status;
      PetscInt  *remote_midnodes, count;

      PetscCallMPI(MPI_Probe(MPI_ANY_SOURCE, 0, PetscObjectComm((PetscObject)pc), &status));
      PetscCallMPI(MPI_Get_count(&status, MPIU_INT, &count));
      PetscCall(PetscMalloc1(count, &remote_midnodes));
      PetscCallMPI(MPI_Recv(remote_midnodes, count, MPIU_INT, status.MPI_SOURCE, 0, PetscObjectComm((PetscObject)pc), MPI_STATUS_IGNORE));

      if (count == 0) continue;

      // We have received some mid node indices, check if they are neighbours of one of our bot nodes
      for (PetscInt k = 0; k < nbot; ++k) {
        for (PetscInt j = ii[botnodes[k]]; j < ii[botnodes[k] + 1]; ++j) {
          for (PetscInt m = 0; m < count; ++m) {
            if (remote_midnodes[m] == colmap[jj[j]]) {
              bot_midnb[k] = 1;
              nbot_midnb++;
              break;
            }
          }
          if (bot_midnb[k] == 1) break;
        }
      }
    }

    

    PetscCall(PetscFree(gmidnodes));
    PetscCall(PetscFree(bot_midnb));
  }

  PetscCall(PetscFree(midnodes));
  PetscCall(PetscFree(intnodes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GaussSeidelKernel(PetscInt i)
{
  (void)i;

  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_TMBSOR(PC pc, Vec b, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)guesszero;
  (void)b;
  (void)y;
  (void)w;

  TMBSOR          tmb = pc->data;
  const PetscInt *idxs;
  PetscInt        nidxs;

  PetscFunctionBeginUser;
  // Receive boundary values from lower processes
  PetscCall(VecScatterBegin(tmb->top->sct, y, tmb->top->vec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(tmb->top->sct, y, tmb->top->vec, INSERT_VALUES, SCATTER_FORWARD));

  // Run Gauss-Seidel on TOP nodes
  PetscCall(ISGetIndices(tmb->top->ids, &idxs));
  PetscCall(ISGetSize(tmb->top->ids, &nidxs));
  for (PetscInt i = 0; i < nidxs; ++i) PetscCall(GaussSeidelKernel(i));
  PetscCall(ISRestoreIndices(tmb->top->ids, &idxs));

  // Send boundary values to lower processors
  PetscCall(VecScatterBegin(tmb->bot_nomid->sct, y, tmb->bot_nomid->vec, INSERT_VALUES, SCATTER_FORWARD));

  // Overlap communication with computation of INT1 nodes
  PetscCall(ISGetIndices(tmb->int1, &idxs));
  PetscCall(ISGetSize(tmb->int1, &nidxs));
  for (PetscInt i = 0; i < nidxs; ++i) PetscCall(GaussSeidelKernel(i));
  PetscCall(ISRestoreIndices(tmb->int1, &idxs));

  // Handle mid nodes

  // Overlap communication with computation of INT2 nodes
  PetscCall(ISGetIndices(tmb->int2, &idxs));
  PetscCall(ISGetSize(tmb->int2, &nidxs));
  for (PetscInt i = 0; i < nidxs; ++i) PetscCall(GaussSeidelKernel(i));
  PetscCall(ISRestoreIndices(tmb->int2, &idxs));

  // Receive boundary values from higher processors
  PetscCall(VecScatterEnd(tmb->bot_nomid->sct, y, tmb->bot_nomid->vec, INSERT_VALUES, SCATTER_FORWARD));

  // Run Gauss-Seide on BOT nodes
  PetscCall(ISGetIndices(tmb->bot_nomid->ids, &idxs));
  PetscCall(ISGetSize(tmb->bot_nomid->ids, &nidxs));
  for (PetscInt i = 0; i < nidxs; ++i) PetscCall(GaussSeidelKernel(i));
  PetscCall(ISRestoreIndices(tmb->bot_nomid->ids, &idxs));

  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_TMBSOR(PC pc)
{
  PetscFunctionBeginUser;
  PetscCall(TMBSOR_ColorProcessors(pc));
  PetscCall(TMBSOR_PartitionNodes(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_TMBSOR(PC pc)
{
  TMBSOR sor;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&sor));
  pc->data = sor;

  pc->ops->setup           = PCSetUp_TMBSOR;
  pc->ops->destroy         = PCDestroy_TMBSOR;
  pc->ops->applyrichardson = PCApplyRichardson_TMBSOR;
  PetscFunctionReturn(PETSC_SUCCESS);
}
