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

#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <adaflo/level_set_okz_compute_curvature.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/level_set_okz_preconditioner.h>
#include <adaflo/level_set_okz_reinitialization.h>
#include <adaflo/util.h>

using VectorType      = LinearAlgebra::distributed::Vector<double>;
using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

static const unsigned int dof_index_ls        = 0;
static const unsigned int dof_index_normal    = 1;
static const unsigned int dof_index_curvature = 2;
static const unsigned int quad_index          = 0;

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

  matrix_free.initialize_dof_vector(ls_solution_update, dof_index_ls);
  matrix_free.initialize_dof_vector(ls_system_rhs, dof_index_ls);
  matrix_free.initialize_dof_vector(curvature_rhs, dof_index_curvature);

  for (unsigned int i = 0; i < dim; ++i)
    matrix_free.initialize_dof_vector(normal_vector_rhs.block(i));

  // TODO
  const double              dt                       = 0.01;
  const unsigned int        stab_steps               = 20;
  std::pair<double, double> last_concentration_range = {-1, +1};
  bool                      first_reinit_step        = true;
  double                    epsilon                  = 1.0;

  AlignedVector<VectorizedArray<double>> cell_diameters;
  double                                 minimal_edge_length;
  double                                 epsilon_used;
  compute_cell_diameters(
    matrix_free, dof_index_ls, cell_diameters, minimal_edge_length, epsilon_used);

  // preconditioner
  DiagonalPreconditioner<double> preconditioner;

  initialize_mass_matrix_diagonal(
    matrix_free, hanging_node_constraints, dof_index_ls, quad_index, preconditioner);

  auto projection_matrix     = std::make_shared<BlockMatrixExtension>();
  auto ilu_projection_matrix = std::make_shared<BlockILUExtension>();

  initialize_projection_matrix(matrix_free,
                               constraints_normals,
                               dof_index_ls,
                               quad_index,
                               epsilon_used,
                               epsilon,
                               cell_diameters,
                               *projection_matrix,
                               *ilu_projection_matrix);

  // normal operator
  LevelSetOKZSolverComputeNormalParameter nomral_parameter;
  nomral_parameter.dof_index_ls            = dof_index_ls;
  nomral_parameter.dof_index_normal        = dof_index_normal;
  nomral_parameter.quad_index              = quad_index;
  nomral_parameter.epsilon                 = epsilon;
  nomral_parameter.approximate_projections = false;

  LevelSetOKZSolverComputeNormal<dim> normal_operator(normal_vector_field,
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
  reinit_parameters.dof_index_ls     = dof_index_ls;
  reinit_parameters.dof_index_normal = dof_index_normal;
  reinit_parameters.quad_index       = quad_index;
  reinit_parameters.do_iteration     = false;

  LevelSetOKZSolverReinitialization<dim> reinit(normal_vector_field,
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
  parameters_curvature.dof_index_ls            = dof_index_ls;
  parameters_curvature.dof_index_curvature     = dof_index_curvature;
  parameters_curvature.dof_index_normal        = dof_index_normal;
  parameters_curvature.quad_index              = quad_index;
  parameters_curvature.epsilon                 = epsilon;
  parameters_curvature.approximate_projections = false;
  parameters_curvature.curvature_correction    = false;

  LevelSetOKZSolverComputeCurvature<dim> curvature_operator(cell_diameters,
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
compute_force_vector_regularized(const MatrixFree<dim, double> &matrix_free,
                                 const VectorType &             ls_solution,
                                 const BlockVectorType &        normal_vector_field,
                                 const VectorType &             curvature_solution,
                                 BlockVectorType &              force_vector)
{
  (void)matrix_free;
  (void)ls_solution;
  (void)normal_vector_field;
  (void)curvature_solution;
  (void)force_vector;
}

template <int dim>
void
compute_force_vector_sharp_interface(const MatrixFree<dim, double> &matrix_free,
                                     const VectorType &             ls_solution,
                                     const BlockVectorType &        normal_vector_field,
                                     const VectorType &             curvature_solution,
                                     BlockVectorType &              force_vector)
{
  (void)matrix_free;
  (void)ls_solution;
  (void)normal_vector_field;
  (void)curvature_solution;
  (void)force_vector;
}


template <int dim>
void
test()
{
  const unsigned int n_global_refinements = 3;
  const unsigned int fe_degree            = 1;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, -1.0, +1.0);
  tria.refine_global(n_global_refinements);

  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  MappingQ1<dim> mapping;

  AffineConstraints<double> constraints, constraints_normals, hanging_node_constraints,
    constraints_curvature;

  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ConstantFunction<dim>(-1.0), constraints);
  constraints.close();

  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ConstantFunction<dim>(0.0), constraints_normals);
  constraints_normals.close();

  VectorTools::interpolate_boundary_values(mapping,
                                           dof_handler,
                                           0,
                                           Functions::ConstantFunction<dim>(0.0),
                                           constraints_curvature);
  constraints_curvature.close();

  hanging_node_constraints.close();

  QGauss<1> quad(fe_degree + 1);

  MatrixFree<dim, double> matrix_free;

  const std::vector<const DoFHandler<dim> *>           dof_handlers{&dof_handler,
                                                          &dof_handler,
                                                          &dof_handler};
  const std::vector<const AffineConstraints<double> *> all_constraints{
    &constraints, &constraints_normals, &constraints_curvature};
  const std::vector<Quadrature<1>> quadratures{quad};
  matrix_free.reinit(mapping, dof_handlers, all_constraints, quadratures);

  // vectors
  BlockVectorType normal_vector_field(dim);
  VectorType      ls_solution;
  VectorType      curvature_solution;
  BlockVectorType force_vector_regularized(dim);
  BlockVectorType force_vector_sharp_interface(dim);

  matrix_free.initialize_dof_vector(ls_solution, dof_index_ls);
  matrix_free.initialize_dof_vector(curvature_solution, dof_index_curvature);

  for (unsigned int i = 0; i < dim; ++i)
    {
      matrix_free.initialize_dof_vector(normal_vector_field.block(i), dof_index_normal);
      matrix_free.initialize_dof_vector(force_vector_regularized.block(i),
                                        dof_index_normal);
      matrix_free.initialize_dof_vector(force_vector_sharp_interface.block(i),
                                        dof_index_normal);
    }

  // initialize level-set

  // compute level-set, normal-vector, and curvature field
  compute_ls_normal_curvature(matrix_free,
                              constraints,
                              constraints_normals,
                              hanging_node_constraints,
                              constraints_curvature,
                              normal_vector_field,
                              ls_solution,
                              curvature_solution);

  //  compute force vector with a regularized approach
  compute_force_vector_regularized(matrix_free,
                                   ls_solution,
                                   normal_vector_field,
                                   curvature_solution,
                                   force_vector_regularized);

  //  compute force vector with a share-interface approach
  compute_force_vector_sharp_interface(matrix_free,
                                       ls_solution,
                                       normal_vector_field,
                                       curvature_solution,
                                       force_vector_sharp_interface);

  // TODO: write computed vectors to Paraview
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test<2>();
}