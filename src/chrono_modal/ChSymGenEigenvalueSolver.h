// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Alessandro Tasora, Dario Mangoni
// =============================================================================

#ifndef CHSYMGENEIGENVALUESOLVER_H
#define CHSYMGENEIGENVALUESOLVER_H

#include "chrono_modal/ChApiModal.h"
#include "chrono_modal/ChGeneralizedEigenvalueSolver.h"
#include "chrono/core/ChMatrix.h"
#include <Eigen/SparseCore>

#include <complex>
#include <functional>

namespace chrono {

// Forward references
class ChDirectSolverLScomplex;
class ChDirectSolverLS;

namespace modal {

/// @addtogroup modal
/// @{

/// Base interface class for iterative eigenvalue solvers for generalized problem with real symmetric matrices.
/// The interfaces assumes shift-and-invert methods.
class ChApiModal ChSymGenEigenvalueSolver : public ChGeneralizedEigenvalueSolver<double> {
  public:
    ChSymGenEigenvalueSolver() {}
    virtual ~ChSymGenEigenvalueSolver(){};

    /// Solve the generalized eigenvalue problem A*eigvects = B*eigvects*diag(eigvals)
    /// A and B are expected to be symmetric and real
    /// 'eigvects' will be resized to [A.rows() x num_modes]
    /// 'eigvals' will be resized to [num_modes]
    /// the number of converged eigenvalues is returned.
    virtual int Solve(const ChSparseMatrix& A,
                      const ChSparseMatrix& B,
                      ChMatrixDynamic<ScalarType>& eigvects,
                      ChVectorDynamic<ScalarType>& eigvals,
                      int num_modes,
                      ScalarType shift) const = 0;

    /// Retrieve the natural frequencies from an eigenvalues array.
    static void GetNaturalFrequencies(const ChVectorDynamic<ScalarType>& eigvals, ChVectorDynamic<double>& freq);

    /// Retrieve the natural frequencies from an eigenvalue.
    static double GetNaturalFrequency(ScalarType eigval) { return std::sqrt(std::abs(eigval)) / CH_2PI; };

    /// Get the optimal shift corresponding to a given frequency.
    static ScalarType GetOptimalShift(double freq) { return -std::pow(freq * CH_2PI, 2); }
};

/// Generalized iterative eigenvalue solver implementing Krylov-Schur shift-and-invert method for real symmetric
/// matrices. Features:
/// - generalized eigenvalue problem
/// - symmetric real sparse matrices
/// - shift-and-invert with real shift
class ChApiModal ChSymGenEigenvalueSolverKrylovSchur : public ChSymGenEigenvalueSolver {
  public:
    ChSymGenEigenvalueSolverKrylovSchur() {}
    virtual ~ChSymGenEigenvalueSolverKrylovSchur(){};

    /// Solve the generalized eigenvalue problem A*eigvects = B*eigvects*diag(eigvals)
    /// A and B are expected to be symmetric and real
    /// 'eigvects' will be resized to [A.rows() x num_modes]
    /// 'eigvals' will be resized to [num_modes]
    /// the number of converged eigenvalues is returned.
    virtual int Solve(const ChSparseMatrix& A,
                      const ChSparseMatrix& B,
                      ChMatrixDynamic<ScalarType>& eigvects,
                      ChVectorDynamic<ScalarType>& eigvals,
                      int num_modes,
                      ScalarType shift) const override;

    /// y = (A - shift_inverse_at * B)^{-1} x by a factorization made elsewhere, taken in place of one of its own by a
    /// solve at that shift: ChModalAssembly's Craig-Bampton reduction hands in its K_IIc's, which its static modes take
    /// too, where the two were factored apiece (6 s each for a mesh of 93390 coordinates). Empty: its own.
    std::function<void(const double* x, double* y)> shift_inverse;
    ScalarType shift_inverse_at = 0;

  private:
    // The iteration, for either shift-and-invert operator
    template <typename OpType, typename BOpType>
    int Run(OpType& op,
            BOpType& Bop,
            ChMatrixDynamic<ScalarType>& eigvects,
            ChVectorDynamic<ScalarType>& eigvals,
            int num_modes,
            int m,
            ScalarType shift) const;
};

/// Generalized iterative eigenvalue solver implementing Lanczos shift-and-invert method for real symmetric matrices.
/// Features:
/// - generalized eigenvalue problem
/// - symmetric real sparse matrices
/// - shift-and-invert with real shift
class ChApiModal ChSymGenEigenvalueSolverLanczos : public ChSymGenEigenvalueSolver {
  public:
    ChSymGenEigenvalueSolverLanczos() {}
    virtual ~ChSymGenEigenvalueSolverLanczos(){};

    /// Solve the generalized eigenvalue problem A*eigvects = B*eigvects*diag(eigvals)
    /// A and B are expected to be symmetric and real
    /// 'eigvects' will be resized to [A.rows() x num_modes]
    /// 'eigvals' will be resized to [num_modes]
    /// the number of converged eigenvalues is returned.
    virtual int Solve(const ChSparseMatrix& A,
                      const ChSparseMatrix& B,
                      ChMatrixDynamic<ScalarType>& eigvects,
                      ChVectorDynamic<ScalarType>& eigvals,
                      int num_modes,
                      ScalarType shift) const override;
};

/// @} modal

}  // end namespace modal
}  // end namespace chrono

#endif
