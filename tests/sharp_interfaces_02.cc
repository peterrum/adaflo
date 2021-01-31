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

#include "sharp_interfaces_util.h"

using VectorType = LinearAlgebra::distributed::Vector<double>;

template <int spacedim>
std::tuple<std::vector<std::pair<int, int>>,
           std::vector<unsigned int>,
           std::vector<Tensor<1, spacedim, double>>,
           std::vector<Point<spacedim>>>
collect_integration_points(
  const Triangulation<spacedim, spacedim> &       tria,
  const Mapping<spacedim, spacedim> &             mapping,
  const std::vector<Point<spacedim>> &            integration_points,
  const std::vector<Tensor<1, spacedim, double>> &integration_values)
{
  std::vector<std::pair<Point<spacedim>, Tensor<1, spacedim, double>>>
    locally_owned_surface_points;

  for (unsigned int i = 0; i < integration_points.size(); ++i)
    locally_owned_surface_points.emplace_back(integration_points[i],
                                              integration_values[i]);

  std::vector<
    std::tuple<Point<spacedim>, Tensor<1, spacedim, double>, std::pair<int, int>>>
    info;

  const std::vector<bool>                    marked_vertices;
  const GridTools::Cache<spacedim, spacedim> cache(tria, mapping);
  const double                               tolerance = 1e-10;
  auto                                       cell_hint = tria.begin_active();

  for (const auto &point_and_weight : locally_owned_surface_points)
    {
      try
        {
          const auto first_cell = GridTools::find_active_cell_around_point(
            cache, point_and_weight.first, cell_hint, marked_vertices, tolerance);

          cell_hint = first_cell.first;

          const auto active_cells_around_point =
            GridTools::find_all_active_cells_around_point(
              mapping, tria, point_and_weight.first, tolerance, first_cell);

          for (const auto &cell_and_reference_coordinate : active_cells_around_point)
            info.emplace_back(
              cell_and_reference_coordinate.second,
              point_and_weight.second,
              std::pair<int, int>(cell_and_reference_coordinate.first->level(),
                                  cell_and_reference_coordinate.first->index()));
        }
      catch (...)
        {}
    }

  // step 4: compress data structures
  std::sort(info.begin(), info.end(), [](const auto &a, const auto &b) {
    return std::get<2>(a) < std::get<2>(b);
  });

  std::vector<std::pair<int, int>>         cells;
  std::vector<unsigned int>                ptrs;
  std::vector<Tensor<1, spacedim, double>> weights;
  std::vector<Point<spacedim>>             points;

  std::pair<int, int> dummy{-1, -1};

  for (const auto &i : info)
    {
      if (dummy != std::get<2>(i))
        {
          dummy = std::get<2>(i);
          cells.push_back(std::get<2>(i));
          ptrs.push_back(weights.size());
        }
      weights.push_back(std::get<1>(i));
      points.push_back(std::get<0>(i));
    }
  ptrs.push_back(weights.size());

  return {cells, ptrs, weights, points};
}



template <int dim, int spacedim>
void
compute_force_vector_sharp_interface(
  const Mapping<dim, spacedim> &   surface_mapping,
  const DoFHandler<dim, spacedim> &surface_dofhandler,
  const DoFHandler<dim, spacedim> &surface_dofhandler_dim,
  const Quadrature<dim> &          surface_quadrature,
  const Mapping<spacedim> &        mapping,
  const DoFHandler<spacedim> &     dof_handler,
  const VectorType &               normal_vector,
  const VectorType &               curvature_vector,
  VectorType &                     force_vector)
{
  std::vector<Point<spacedim>>             integration_points;
  std::vector<Tensor<1, spacedim, double>> integration_values;

  {
    FEValues<dim, spacedim> fe_eval(surface_mapping,
                                    surface_dofhandler.get_fe(),
                                    surface_quadrature,
                                    update_values | update_quadrature_points |
                                      update_JxW_values);
    FEValues<dim, spacedim> fe_eval_dim(surface_mapping,
                                        surface_dofhandler_dim.get_fe(),
                                        surface_quadrature,
                                        update_values);

    const auto &tria_surface = surface_dofhandler.get_triangulation();

    for (const auto &cell : tria_surface.active_cell_iterators())
      {
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell(&tria_surface,
                                                                     cell->level(),
                                                                     cell->index(),
                                                                     &surface_dofhandler);
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
          &tria_surface, cell->level(), cell->index(), &surface_dofhandler_dim);

        fe_eval.reinit(dof_cell);
        fe_eval_dim.reinit(dof_cell_dim);

        std::vector<double>         curvature_values(fe_eval.dofs_per_cell);
        std::vector<Vector<double>> normal_values(fe_eval.dofs_per_cell,
                                                  Vector<double>(spacedim));

        fe_eval.get_function_values(curvature_vector, curvature_values);
        fe_eval_dim.get_function_values(normal_vector, normal_values);

        for (const auto q : fe_eval_dim.quadrature_point_indices())
          {
            Tensor<1, spacedim, double> result;
            for (unsigned int i = 0; i < spacedim; ++i)
              result[i] = curvature_values[q] * normal_values[q][i] * fe_eval.JxW(q);

            integration_points.push_back(fe_eval.quadrature_point(q));
            integration_values.push_back(result);
          }
      }
  }

  const auto [cells, ptrs, weights, points] = collect_integration_points(
    dof_handler.get_triangulation(), mapping, integration_points, integration_values);

  AffineConstraints<double> constraints; // TODO: use the right ones

  FEPointEvaluation<spacedim, spacedim> phi_normal_force(mapping, dof_handler.get_fe());

  std::vector<double>                  buffer;
  std::vector<types::global_dof_index> local_dof_indices;

  for (unsigned int i = 0; i < cells.size(); ++i)
    {
      typename DoFHandler<spacedim>::active_cell_iterator cell = {
        &dof_handler.get_triangulation(), cells[i].first, cells[i].second, &dof_handler};

      const unsigned int n_dofs_per_cell = cell->get_fe().n_dofs_per_cell();

      local_dof_indices.resize(n_dofs_per_cell);
      buffer.resize(n_dofs_per_cell);

      cell->get_dof_indices(local_dof_indices);

      const unsigned int n_points = ptrs[i + 1] - ptrs[i];

      const ArrayView<const Point<spacedim>> unit_points(points.data() + ptrs[i],
                                                         n_points);
      const ArrayView<const Tensor<1, spacedim, double>> JxW(weights.data() + ptrs[i],
                                                             n_points);

      for (unsigned int q = 0; q < n_points; ++q)
        phi_normal_force.submit_value(JxW[q], q);

      phi_normal_force.integrate(cell, unit_points, buffer, EvaluationFlags::values);

      constraints.distribute_local_to_global(buffer, local_dof_indices, force_vector);
    }
}



template <int dim>
void
test()
{
  const unsigned int spacedim = dim + 1;

  const unsigned int fe_degree      = 3;
  const unsigned int mapping_degree = fe_degree;
  const unsigned int n_refinements  = 5;

  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_sphere(tria, Point<spacedim>(), 0.5);
  tria.refine_global(n_refinements);

  // quadrature rule and FE for curvature
  FE_Q<dim, spacedim>       fe(fe_degree);
  QGaussLobatto<dim>        quadrature(fe_degree + 1);
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // FE for normal
  FESystem<dim, spacedim>   fe_dim(fe, spacedim);
  DoFHandler<dim, spacedim> dof_handler_dim(tria);
  dof_handler_dim.distribute_dofs(fe_dim);

  // Set up MappingFEField
  Vector<double> euler_vector(dof_handler_dim.n_dofs());
  VectorTools::get_position_vector(dof_handler_dim,
                                   euler_vector,
                                   MappingQGeneric<dim, spacedim>(mapping_degree));
  MappingFEField<dim, spacedim> mapping(dof_handler_dim, euler_vector);


  // compute normal vector
  VectorType normal_vector(dof_handler_dim.n_dofs());
  {
    FEValues<dim, spacedim> fe_eval_dim(mapping,
                                        fe_dim,
                                        fe_dim.get_unit_support_points(),
                                        update_normal_vectors | update_gradients);

    Vector<double> normal_temp;

    for (const auto &cell : tria.active_cell_iterators())
      {
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
          &tria, cell->level(), cell->index(), &dof_handler_dim);

        fe_eval_dim.reinit(dof_cell_dim);

        normal_temp.reinit(fe_eval_dim.dofs_per_cell);
        normal_temp = 0.0;

        for (const auto q : fe_eval_dim.quadrature_point_indices())
          {
            const auto normal = fe_eval_dim.normal_vector(q);

            const unsigned int comp =
              dof_handler_dim.get_fe().system_to_component_index(q).first;

            normal_temp[q] = normal[comp];
          }

        dof_cell_dim->set_dof_values(normal_temp, normal_vector);
      }
  }

  // compute curvature
  VectorType curvature_vector(dof_handler.n_dofs());
  {
    FEValues<dim, spacedim> fe_eval(mapping, fe, quadrature, update_gradients);
    FEValues<dim, spacedim> fe_eval_dim(
      mapping, fe_dim, quadrature, update_values | update_gradients | update_JxW_values);

    Vector<double> curvature_temp;

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

        std::vector<Vector<double>> normal_values(fe_eval.dofs_per_cell,
                                                  Vector<double>(spacedim));

        std::vector<std::vector<Tensor<1, spacedim, double>>> normal_gradients(
          fe_eval.dofs_per_cell, std::vector<Tensor<1, spacedim, double>>(spacedim));

        fe_eval_dim.get_function_values(normal_vector, normal_values);
        fe_eval_dim.get_function_gradients(normal_vector, normal_gradients);

        for (const auto q : fe_eval_dim.quadrature_point_indices())
          {
            double curvature = 0.0;

            for (unsigned c = 0; c < spacedim; ++c)
              curvature += normal_gradients[q][c][c];

            curvature_temp[q] = curvature;

            Tensor<1, spacedim, double> result;
            for (unsigned int i = 0; i < spacedim; ++i)
              result[i] = curvature * normal_values[q][i] * fe_eval.JxW(q);
            std::cout << result << std::endl;
          }

        dof_cell->set_dof_values(curvature_temp, curvature_vector);
      }
  }

  const unsigned int background_n_global_refinements = 6;
  const unsigned int background_fe_degree            = 1;

  Triangulation<spacedim> background_tria;
  GridGenerator::hyper_cube(background_tria, -1.0, +1.0);
  background_tria.refine_global(background_n_global_refinements);

  FESystem<spacedim>   background_fe(FE_Q<spacedim>{background_fe_degree}, spacedim);
  DoFHandler<spacedim> background_dof_handler(background_tria);
  background_dof_handler.distribute_dofs(background_fe);

  MappingQ1<spacedim> background_mapping;

  VectorType force_vector_sharp_interface(background_dof_handler.n_dofs());

  compute_force_vector_sharp_interface(mapping,
                                       dof_handler,
                                       dof_handler_dim,
                                       QGauss<dim>(fe_degree + 1),
                                       background_mapping,
                                       background_dof_handler,
                                       normal_vector,
                                       curvature_vector,
                                       force_vector_sharp_interface);

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
    data_out.write_vtu_with_pvtu_record("./",
                                        "sharp_interface_02_surface",
                                        0,
                                        MPI_COMM_WORLD);
  }

  {
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;

    DataOut<spacedim> data_out;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(background_dof_handler);
    data_out.add_data_vector(background_dof_handler,
                             force_vector_sharp_interface,
                             "force");

    data_out.build_patches(background_mapping, background_fe_degree + 1);
    data_out.write_vtu_with_pvtu_record("./",
                                        "sharp_interface_02_background",
                                        0,
                                        MPI_COMM_WORLD);
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