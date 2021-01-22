// --------------------------------------------------------------------------
//
// Copyright (C) 2021 by the adaflo authors
//
// This file is part of the adaflo library.
//
// The adaflo library is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.  The full text of the
// license can be found in the file LICENSE at the top level of the adaflo
// distribution.
//
// --------------------------------------------------------------------------

#include <adaflo/level_set_okz_compute_curvature.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/level_set_okz_preconditioner.h>
#include <adaflo/level_set_okz_reinitialization.h>
#include <adaflo/util.h>

using VectorType      = LinearAlgebra::distributed::Vector<double>;
using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

template <int dim>
void
compute_ls_normal_curvature(const MatrixFree<dim, double> &  matrix_free,
                            const AffineConstraints<double> &constraints,
                            const AffineConstraints<double> &constraints_normals,
                            const AffineConstraints<double> &hanging_node_constraints,
                            const AffineConstraints<double> &constraints_curvature,
                            BlockVectorType &                normal_vector_field,
                            VectorType &                     ls_solution,
                            VectorType &                     curvature_solution)
{
  //
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  // vectors
  BlockVectorType normal_vector_rhs(dim);
  VectorType      ls_solution_update;
  VectorType      ls_system_rhs;
  VectorType      curvature_rhs;

  // TODO
  const double              dt                       = 0.01;
  const unsigned int        stab_steps               = 20;
  std::pair<double, double> last_concentration_range = {-1, +1};
  bool                      first_reinit_step        = true;
  double                    epsilon; // TODO

  AlignedVector<VectorizedArray<double>> cell_diameters;
  double                                 minimal_edge_length;
  double                                 epsilon_used;
  compute_cell_diameters(
    matrix_free, 0 /*TODO*/, cell_diameters, minimal_edge_length, epsilon_used);

  // preconditioner
  DiagonalPreconditioner<double> preconditioner;

  initialize_mass_matrix_diagonal(
    matrix_free, hanging_node_constraints, 2, 2, preconditioner);

  auto projection_matrix     = std::make_shared<BlockMatrixExtension>();
  auto ilu_projection_matrix = std::make_shared<BlockILUExtension>();

  initialize_projection_matrix(matrix_free,
                               constraints_normals,
                               2 /*TODO*/,
                               2 /*TODO*/,
                               epsilon_used,
                               epsilon,
                               cell_diameters,
                               *projection_matrix,
                               *ilu_projection_matrix);

  // normal operator
  LevelSetOKZSolverComputeNormalParameter nomral_parameter;
  LevelSetOKZSolverComputeNormal<dim>     normal_operator(normal_vector_field,
                                                      normal_vector_rhs,
                                                      ls_solution,
                                                      cell_diameters,
                                                      epsilon_used,
                                                      minimal_edge_length,
                                                      constraints_normals,
                                                      nomral_parameter,
                                                      matrix_free,
                                                      preconditioner,
                                                      projection_matrix,
                                                      ilu_projection_matrix);

  // reinitialization operator
  LevelSetOKZSolverReinitializationParameter reinit_parameters;
  LevelSetOKZSolverReinitialization<dim>     reinit(normal_vector_field,
                                                cell_diameters,
                                                epsilon_used,
                                                minimal_edge_length,
                                                constraints,
                                                ls_solution_update,
                                                ls_solution,
                                                ls_system_rhs,
                                                pcout,
                                                preconditioner,
                                                last_concentration_range,
                                                reinit_parameters,
                                                first_reinit_step,
                                                matrix_free);

  // curvature operator
  LevelSetOKZSolverComputeCurvatureParameter parameters_curvature;
  LevelSetOKZSolverComputeCurvature<dim>     curvature_operator(cell_diameters,
                                                            normal_vector_field,
                                                            constraints_curvature,
                                                            constraints,
                                                            epsilon_used,
                                                            curvature_rhs,
                                                            parameters_curvature,
                                                            curvature_solution,
                                                            ls_solution,
                                                            matrix_free,
                                                            preconditioner,
                                                            projection_matrix,
                                                            ilu_projection_matrix);

  // set initial condition

  // perform reinitialization
  reinit.reinitialize(dt, stab_steps, 0, [&normal_operator](const bool fast) {
    normal_operator.compute_normal(fast);
  });

  // compute normal vectors
  normal_operator.compute_normal(false);

  // compute curvature
  curvature_operator.compute_curvature();
}


template <int dim>
void
test()
{
  AffineConstraints<double> constraints, constraints_normals, hanging_node_constraints,
    constraints_curvature;
  MatrixFree<dim, double> matrix_free;

  // vectors
  BlockVectorType normal_vector_field(dim);
  VectorType      ls_solution;
  VectorType      curvature_solution;

  compute_ls_normal_curvature(matrix_free,
                              constraints,
                              constraints_normals,
                              hanging_node_constraints,
                              constraints_curvature,
                              normal_vector_field,
                              ls_solution,
                              curvature_solution);
}



int
main()
{
  test<2>();
}