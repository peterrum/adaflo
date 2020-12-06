// --------------------------------------------------------------------------
//
// Copyright (C) 2020 by the adaflo authors
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

#include <deal.II/base/timer.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <adaflo/level_set_okz_reinitialization.h>
#include <adaflo/level_set_okz_template_instantations.h>

using namespace dealii;

template <int dim>
template <int ls_degree, bool diffuse_only>
void
LevelSetOKZSolverReinitialization<dim>::local_reinitialize(
  const MatrixFree<dim, double> &                   data,
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  const double dtau_inv = std::max(0.95 / (1. / (dim * dim) * this->minimal_edge_length /
                                           this->parameters.concentration_subdivisions),
                                   1. / (5. * this->time_stepping.step_size()));

  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function)
  FEEvaluation<dim, ls_degree, 2 * ls_degree> phi(data, 2, 2);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(src);
      phi.evaluate(true, true, false);

      VectorizedArray<double> cell_diameter = this->cell_diameters[cell];
      VectorizedArray<double> diffusion =
        std::max(make_vectorized_array(this->epsilon_used),
                 cell_diameter / static_cast<double>(ls_degree));

      const Tensor<1, dim, VectorizedArray<double>> *normal =
        &evaluated_convection[cell * phi.n_q_points];
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        if (!diffuse_only)
          {
            phi.submit_value(dtau_inv * phi.get_value(q), q);
            phi.submit_gradient((diffusion * (normal[q] * phi.get_gradient(q))) *
                                  normal[q],
                                q);
          }
        else
          {
            phi.submit_value(dtau_inv * phi.get_value(q), q);
            phi.submit_gradient(phi.get_gradient(q) * diffusion, q);
          }

      phi.integrate(true, true);
      phi.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int ls_degree, bool diffuse_only>
void
LevelSetOKZSolverReinitialization<dim>::local_reinitialize_rhs(
  const MatrixFree<dim, double> &             data,
  LinearAlgebra::distributed::Vector<double> &dst,
  const LinearAlgebra::distributed::Vector<double> &,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function) and 4 means constraints_normals
  FEEvaluation<dim, ls_degree, 2 * ls_degree>      phi(data, 2, 2);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, dim> normals(data, 4, 2);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values_plain(this->solution.block(0));
      phi.evaluate(true, true, false);

      normals.reinit(cell);
      normals.read_dof_values_plain(this->normal_vector_field);
      normals.evaluate(true, false, false);

      VectorizedArray<double> cell_diameter = this->cell_diameters[cell];
      VectorizedArray<double> diffusion =
        std::max(make_vectorized_array(this->epsilon_used),
                 cell_diameter / static_cast<double>(ls_degree));

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        if (!diffuse_only)
          {
            Tensor<1, dim, VectorizedArray<double>> grad = phi.get_gradient(q);
            if (first_reinit_step)
              {
                Tensor<1, dim, VectorizedArray<double>> normal = normals.get_value(q);
                normal /= std::max(make_vectorized_array(1e-4), normal.norm());
                evaluated_convection[cell * phi.n_q_points + q] = normal;
              }
            // take normal as it was for the first reinit step
            Tensor<1, dim, VectorizedArray<double>> normal =
              evaluated_convection[cell * phi.n_q_points + q];
            phi.submit_gradient(normal *
                                  (0.5 * (1. - phi.get_value(q) * phi.get_value(q)) -
                                   (normal * grad * diffusion)),
                                q);
          }
        else
          {
            phi.submit_gradient(-diffusion * phi.get_gradient(q), q);
          }

      phi.integrate(false, true);
      phi.distribute_local_to_global(dst);
    }
}



template <int dim>
void
LevelSetOKZSolverReinitialization<dim>::reinitialization_vmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const bool                                        diffuse_only) const
{
  dst = 0.;
  if (diffuse_only)
    {
#define OPERATION(c_degree, u_degree)                                              \
  this->matrix_free.cell_loop(&LevelSetOKZSolverReinitialization<                  \
                                dim>::template local_reinitialize<c_degree, true>, \
                              this,                                                \
                              dst,                                                 \
                              src)

      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }
  else
    {
#define OPERATION(c_degree, u_degree)                                               \
  this->matrix_free.cell_loop(&LevelSetOKZSolverReinitialization<                   \
                                dim>::template local_reinitialize<c_degree, false>, \
                              this,                                                 \
                              dst,                                                  \
                              src)

      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

  for (unsigned int i = 0; i < this->matrix_free.get_constrained_dofs(2).size(); ++i)
    dst.local_element(this->matrix_free.get_constrained_dofs(2)[i]) =
      preconditioner.get_vector().local_element(
        this->matrix_free.get_constrained_dofs(2)[i]) *
      src.local_element(this->matrix_free.get_constrained_dofs(2)[i]);
}



template <int dim>
struct ReinitializationMatrix
{
  ReinitializationMatrix(const LevelSetOKZSolverReinitialization<dim> &problem,
                         const bool                                    diffuse_only)
    : problem(problem)
    , diffuse_only(diffuse_only)
  {}

  void
  vmult(LinearAlgebra::distributed::Vector<double> &      dst,
        const LinearAlgebra::distributed::Vector<double> &src) const
  {
    problem.reinitialization_vmult(dst, src, diffuse_only);
  }

  const LevelSetOKZSolverReinitialization<dim> &problem;
  const bool                                    diffuse_only;
};



template <int dim>
void
LevelSetOKZSolverReinitialization<dim>::reinitialize(const unsigned int stab_steps,
                                                     const unsigned int diff_steps,
                                                     const bool)
{
  // This function assembles and solves for a given profile using the approach
  // described in the paper by Olsson, Kreiss, and Zahedi.

  std::cout.precision(3);

  // perform several reinitialization steps until we reach the maximum number
  // of steps.
  //
  // TODO: make an adaptive choice of the number of iterations
  unsigned actual_diff_steps = diff_steps;
  if (this->last_concentration_range.first < -1.02 ||
      this->last_concentration_range.second > 1.02)
    actual_diff_steps += 3;
  if (!this->parameters.do_iteration)
    this->pcout << (this->time_stepping.now() == this->time_stepping.start() ? "  " :
                                                                               " and ")
                << "reinitialize (";
  for (unsigned int tau = 0; tau < actual_diff_steps + stab_steps; tau++)
    {
      first_reinit_step = (tau == actual_diff_steps);
      if (first_reinit_step)
        compute_normal(true);

      TimerOutput::Scope timer(*this->timer, "LS reinitialization step.");

      // compute right hand side
      LinearAlgebra::distributed::Vector<double> &rhs = this->system_rhs.block(0);
      LinearAlgebra::distributed::Vector<double> &increment =
        this->solution_update.block(0);
      rhs = 0;

      if (tau < actual_diff_steps)
        {
#define OPERATION(c_degree, u_degree)                                                  \
  this->matrix_free.cell_loop(&LevelSetOKZSolverReinitialization<                      \
                                dim>::template local_reinitialize_rhs<c_degree, true>, \
                              this,                                                    \
                              rhs,                                                     \
                              this->solution.block(0))

          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
        }
      else
        {
#define OPERATION(c_degree, u_degree)                                                   \
  this->matrix_free.cell_loop(&LevelSetOKZSolverReinitialization<                       \
                                dim>::template local_reinitialize_rhs<c_degree, false>, \
                              this,                                                     \
                              rhs,                                                      \
                              this->solution.block(0))

          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
        }

      // solve linear system
      {
        ReinitializationMatrix<dim> matrix(*this, tau < actual_diff_steps);
        increment = 0;

        // reduce residual by 1e-6. To obtain good interface shapes, it is
        // essential that this tolerance is relative to the rhs
        // (ReductionControl steered solver, last argument determines the
        // solver)
        ReductionControl solver_control(2000, 1e-50, 1e-6);
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        cg.solve(matrix, increment, rhs, preconditioner);
        this->constraints.distribute(increment);
        if (!this->parameters.do_iteration)
          {
            if (tau < actual_diff_steps)
              this->pcout << "d" << solver_control.last_step();
            else
              this->pcout << solver_control.last_step();
          }
      }

      this->solution.block(0) += increment;
      this->solution.block(0).update_ghost_values();

      // check residual
      const double update_norm = increment.l2_norm();
      if (update_norm < 1e-6)
        break;

      if (!this->parameters.do_iteration && tau < actual_diff_steps + stab_steps - 1)
        this->pcout << " + ";
    }

  if (!this->parameters.do_iteration)
    this->pcout << ")" << std::endl << std::flush;
}

template class LevelSetOKZSolverReinitialization<2>;
template class LevelSetOKZSolverReinitialization<3>;
