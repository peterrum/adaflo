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

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_point_evaluation.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <adaflo/level_set_okz_compute_curvature.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/level_set_okz_preconditioner.h>
#include <adaflo/level_set_okz_reinitialization.h>
#include <adaflo/util.h>


template <int dim>
void
test()
{
  const unsigned int spacedim = dim + 1;

  const unsigned int fe_degree      = 3;
  const unsigned int mapping_degree = fe_degree;
  const unsigned int n_refinements  = 2;

  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_sphere(tria, Point<spacedim>(), 0.5);
  tria.refine_global(n_refinements);

  // quadrature rule
  QGauss<dim> quadrature(fe_degree + 1);

  // FE for curvature
  FE_DGQArbitraryNodes<dim, spacedim> fe(quadrature);
  DoFHandler<dim, spacedim>           dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // FE for normal
  FESystem<dim, spacedim>   fe_dim(fe, spacedim);
  DoFHandler<dim, spacedim> dof_handler_dim(tria);
  dof_handler_dim.distribute_dofs(fe_dim);

  // Euler vector for MappingFEField
  Vector<double> euler_vector(dof_handler_dim.n_dofs());

  // setup Euler vector (like VectorTools::get_position_vector() but taking a
  // mapping)
  {
    MappingQGeneric<dim, spacedim> mapping(mapping_degree);

    FEValues<dim, spacedim> fe_eval(mapping,
                                    fe_dim,
                                    quadrature,
                                    update_quadrature_points);

    Vector<double> temp;

    for (const auto &cell : dof_handler_dim.active_cell_iterators())
      {
        fe_eval.reinit(cell);

        AssertDimension(fe_eval.n_quadrature_points * spacedim, fe_eval.dofs_per_cell);

        temp.reinit(fe_eval.dofs_per_cell);

        for (const auto q : fe_eval.quadrature_point_indices())
          {
            const auto point = fe_eval.quadrature_point(q);

            for (unsigned int c = 0; c < spacedim; ++c)
              temp[fe_eval.n_quadrature_points * c + q] = point[c];
          }

        cell->set_dof_values(temp, euler_vector);
      }
  }

  Vector<double>                normal_vector(dof_handler_dim.n_dofs());
  Vector<double>                curvature_vector(dof_handler.n_dofs());
  MappingFEField<dim, spacedim> mapping(dof_handler_dim, euler_vector);

  {
    FEValues<dim, spacedim> fe_eval(mapping, fe, quadrature, update_gradients);
    FEValues<dim, spacedim> fe_eval_dim(mapping,
                                        fe_dim,
                                        quadrature,
                                        update_normal_vectors | update_gradients);

    Vector<double> normal_temp;
    Vector<double> curvature_temp;

    // compute normal vector
    for (const auto &cell : tria.active_cell_iterators())
      {
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
          &tria, cell->level(), cell->index(), &dof_handler_dim);

        fe_eval_dim.reinit(dof_cell_dim);

        AssertDimension(fe_eval_dim.n_quadrature_points * spacedim,
                        fe_eval_dim.dofs_per_cell);

        normal_temp.reinit(fe_eval_dim.dofs_per_cell);
        normal_temp = 0.0;

        for (const auto q : fe_eval_dim.quadrature_point_indices())
          {
            const auto normal = fe_eval_dim.normal_vector(q);

            for (unsigned int c = 0; c < spacedim; ++c)
              {
                normal_temp[fe_eval_dim.n_quadrature_points * c + q] = normal[c];
              }
          }

        dof_cell_dim->set_dof_values(normal_temp, normal_vector);
      }

    // compute curvature
    for (const auto &cell : tria.active_cell_iterators())
      {
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell(&tria,
                                                                     cell->level(),
                                                                     cell->index(),
                                                                     &dof_handler);
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
          &tria, cell->level(), cell->index(), &dof_handler_dim);

        fe_eval.reinit(dof_cell);
        fe_eval_dim.reinit(dof_cell_dim);

        curvature_temp.reinit(fe_eval.dofs_per_cell);

        std::vector<std::vector<Tensor<1, spacedim, double>>> normal_gradients(
          fe_eval.dofs_per_cell, std::vector<Tensor<1, spacedim, double>>(spacedim));

        fe_eval_dim.get_function_gradients(normal_vector, normal_gradients);

        for (const auto q : fe_eval_dim.quadrature_point_indices())
          {
            double temp = 0.0;

            for (unsigned c = 0; c < spacedim; ++c)
              temp += normal_gradients[q][c][c];

            curvature_temp[q] += temp;
          }

        dof_cell->set_dof_values(curvature_temp, curvature_vector);
      }
  }

  // write computed vectors to Paraview
  {
    DataOutBase::VtkFlags flags;
    // flags.write_higher_order_cells = true;

    DataOut<dim, DoFHandler<dim, spacedim>> data_out;
    data_out.set_flags(flags);
    data_out.add_data_vector(dof_handler, curvature_vector, "curvature");
    data_out.add_data_vector(dof_handler_dim, normal_vector, "normal");

    data_out.build_patches(
      mapping,
      fe_degree + 1,
      DataOut<dim, DoFHandler<dim, spacedim>>::CurvedCellRegion::curved_inner_cells);
    data_out.write_vtu_with_pvtu_record("./", "result", 0, MPI_COMM_WORLD);
  }

  {
    curvature_vector.print(std::cout);
  }
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test<1>();
}