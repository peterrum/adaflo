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

#include <deal.II/fe/fe_point_evaluation.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

namespace dealii
{
  namespace VectorTools
  {
    template <int dim, int spacedim, typename VectorType>
    void
    get_position_vector(const DoFHandler<dim, spacedim> &dof_handler_dim,
                        VectorType &                     euler_vector,
                        const Mapping<dim, spacedim> &   mapping)
    {
      FEValues<dim, spacedim> fe_eval(
        mapping,
        dof_handler_dim.get_fe(),
        Quadrature<dim>(dof_handler_dim.get_fe().get_unit_support_points()),
        update_quadrature_points);

      Vector<double> temp;

      for (const auto &cell : dof_handler_dim.active_cell_iterators())
        {
          fe_eval.reinit(cell);

          temp.reinit(fe_eval.dofs_per_cell);

          for (const auto q : fe_eval.quadrature_point_indices())
            {
              const auto point = fe_eval.quadrature_point(q);

              const unsigned int comp =
                dof_handler_dim.get_fe().system_to_component_index(q).first;

              temp[q] = point[comp];
            }

          cell->set_dof_values(temp, euler_vector);
        }
    }

    template <int dim, int spacedim, typename VectorType>
    void
    update_position_vector(const double                          dt,
                           const DoFHandler<spacedim, spacedim> &background_dofhandler,
                           const Mapping<spacedim, spacedim> &   background_mapping,
                           const VectorType &                    velocity_vector,
                           const DoFHandler<dim, spacedim> &     euler_dofhandler,
                           const Mapping<dim, spacedim> &        euler_mapping,
                           VectorType &                          euler_vector)
    {
      FEValues<dim, spacedim> fe_eval(
        euler_mapping,
        euler_dofhandler.get_fe(),
        Quadrature<dim>(euler_dofhandler.get_fe().get_unit_support_points()),
        update_quadrature_points);

      Vector<double>                       temp;
      std::vector<types::global_dof_index> temp_dof_indices;

      auto euler_vector_temp = euler_vector;
      auto euler_vector_bool = euler_vector;
      euler_vector_bool      = 0.0;

      const std::vector<bool>                    marked_vertices;
      const GridTools::Cache<spacedim, spacedim> cache(
        background_dofhandler.get_triangulation(), background_mapping);
      const double tolerance = 1e-10;
      auto         cell_hint = background_dofhandler.get_triangulation().begin_active();

      for (const auto &cell : euler_dofhandler.active_cell_iterators())
        {
          fe_eval.reinit(cell);

          temp.reinit(fe_eval.dofs_per_cell);
          temp_dof_indices.resize(fe_eval.dofs_per_cell);

          cell->get_dof_indices(temp_dof_indices);
          cell->get_dof_values(euler_vector, temp);

          for (const auto q : fe_eval.quadrature_point_indices())
            {
              // if (euler_vector_bool[temp_dof_indices[q]] == 1.0)
              //  continue;

              euler_vector_bool[temp_dof_indices[q]] = 1.0;

              const auto velocity = [&] {
                const auto cell_and_reference_coordinate =
                  GridTools::find_active_cell_around_point(cache,
                                                           fe_eval.quadrature_point(q),
                                                           cell_hint,
                                                           marked_vertices,
                                                           tolerance);

                std::vector<double> buffer(
                  background_dofhandler.get_fe().n_dofs_per_cell());

                typename DoFHandler<spacedim, spacedim>::active_cell_iterator
                  background_cell{&background_dofhandler.get_triangulation(),
                                  cell_and_reference_coordinate.first->level(),
                                  cell_and_reference_coordinate.first->index(),
                                  &background_dofhandler};
                background_cell->get_dof_values(velocity_vector,
                                                buffer.begin(),
                                                buffer.end());

                const ArrayView<const Point<spacedim>> unit_points(
                  &cell_and_reference_coordinate.second, 1);

                FEPointEvaluation<spacedim, spacedim> phi_velocity(
                  background_mapping, background_dofhandler.get_fe());
                phi_velocity.evaluate(cell_and_reference_coordinate.first,
                                      unit_points,
                                      buffer,
                                      EvaluationFlags::values);

                return phi_velocity.get_value(0);
              }();

              const unsigned int comp =
                euler_dofhandler.get_fe().system_to_component_index(q).first;

              // temp[q] += dt * velocity[comp];
              temp[q] = fe_eval.quadrature_point(q)[comp] + dt * velocity[comp];
            }

          cell->set_dof_values(temp, euler_vector_temp);
        }

      euler_vector = euler_vector_temp;
    }
  } // namespace VectorTools

  namespace GridTools
  {
    template <int dim, int spacedim>
    void
    construct_polygon(
      const Mapping<dim, spacedim> &   mapping,
      const DoFHandler<dim, spacedim> &dof_handler,
      boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double>> &poly)
    {
      typedef boost::geometry::model::d2::point_xy<double> point_type;
      typedef boost::geometry::model::polygon<
        boost::geometry::model::d2::point_xy<double>>
        polygon_type;

      std::vector<boost::geometry::model::d2::point_xy<double>> points;
      {
        FEValues<dim, spacedim> fe_eval(mapping,
                                        dof_handler.get_fe(),
                                        QGauss<dim>(dof_handler.get_fe().degree + 1),
                                        update_quadrature_points);

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            fe_eval.reinit(cell);

            for (const auto q : fe_eval.quadrature_point_indices())
              {
                const auto point = fe_eval.quadrature_point(q);
                points.emplace_back(point[0], point[1]);
              }
          }
      }

      points.push_back(points.front());

      boost::geometry::assign_points(poly, points);
    }


    template <int dim>
    double
    within(
      const boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double>>
        &               polygon,
      const Point<dim> &point)
    {
      boost::geometry::model::d2::point_xy<double> p(point[0], point[1]);
      return boost::geometry::within(p, polygon);
    }
    template <int dim>
    VectorizedArray<double>
    within(
      const boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double>>
        &                                        polygon,
      const Point<dim, VectorizedArray<double>> &points)
    {
      VectorizedArray<double> result;

      for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
        {
          boost::geometry::model::d2::point_xy<double> p(points[0][v], points[1][v]);
          result[v] = boost::geometry::within(p, polygon);
        }

      return result;
    }

    template <int dim, int spacedim, typename VectorType>
    void
    within(const MappingFEField<dim, spacedim> &mapping,
           const DoFHandler<dim, spacedim> &    dof_handler,
           const Mapping<spacedim> &            background_mapping,
           const DoFHandler<spacedim> &         background_dof_handler,
           VectorType &                         force_vector_sharp_interface)
    {
      typedef boost::geometry::model::d2::point_xy<double> point_type;
      typedef boost::geometry::model::polygon<
        boost::geometry::model::d2::point_xy<double>>
        polygon_type;

      polygon_type poly;

      construct_polygon(mapping, dof_handler, poly);

      {
        FEValues<spacedim> fe_eval(
          background_mapping,
          background_dof_handler.get_fe(),
          background_dof_handler.get_fe().get_unit_support_points(),
          update_quadrature_points);

        for (const auto &cell : background_dof_handler.active_cell_iterators())
          {
            fe_eval.reinit(cell);

            Vector<double> vec(fe_eval.n_quadrature_points);

            for (const auto q : fe_eval.quadrature_point_indices())
              {
                vec[q] = static_cast<double>(within(poly, fe_eval.quadrature_point(q)));
              }

            cell->set_dof_values(vec, force_vector_sharp_interface);
          }
      }
    }
  } // namespace GridTools



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



  template <int dim, int spacedim, typename VectorType>
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
          TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell(
            &tria_surface, cell->level(), cell->index(), &surface_dofhandler);
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
          &dof_handler.get_triangulation(),
          cells[i].first,
          cells[i].second,
          &dof_handler};

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

} // namespace dealii
