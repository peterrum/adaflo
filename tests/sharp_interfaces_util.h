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
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <adaflo/level_set_okz_advance_concentration.h>
#include <adaflo/level_set_okz_compute_curvature.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/level_set_okz_preconditioner.h>
#include <adaflo/level_set_okz_reinitialization.h>
#include <adaflo/util.h>

#include <filesystem>

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
    const double                     surface_tension,
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
                result[i] = -curvature_values[q] * normal_values[q][i] * fe_eval.JxW(q) *
                            surface_tension;

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



  template <int dim, int spacedim, typename VectorType>
  void
  compute_normal(const Mapping<dim, spacedim> &   mapping,
                 const DoFHandler<dim, spacedim> &dof_handler_dim,
                 VectorType &                     normal_vector)
  {
    FEValues<dim, spacedim> fe_eval_dim(
      mapping,
      dof_handler_dim.get_fe(),
      dof_handler_dim.get_fe().get_unit_support_points(),
      update_normal_vectors | update_gradients);

    Vector<double> normal_temp;

    for (const auto &cell : dof_handler_dim.get_triangulation().active_cell_iterators())
      {
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
          &dof_handler_dim.get_triangulation(),
          cell->level(),
          cell->index(),
          &dof_handler_dim);

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



  template <int dim, int spacedim, typename VectorType>
  void
  compute_curvature(const Mapping<dim, spacedim> &   mapping,
                    const DoFHandler<dim, spacedim> &dof_handler_dim,
                    const DoFHandler<dim, spacedim> &dof_handler,
                    const Quadrature<dim>            quadrature,
                    const VectorType &               normal_vector,
                    VectorType &                     curvature_vector)
  {
    FEValues<dim, spacedim> fe_eval(mapping,
                                    dof_handler.get_fe(),
                                    quadrature,
                                    update_gradients);
    FEValues<dim, spacedim> fe_eval_dim(mapping,
                                        dof_handler_dim.get_fe(),
                                        quadrature,
                                        update_gradients);

    Vector<double> curvature_temp;

    for (const auto &cell : dof_handler.get_triangulation().active_cell_iterators())
      {
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell(
          &dof_handler.get_triangulation(), cell->level(), cell->index(), &dof_handler);
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
          &dof_handler_dim.get_triangulation(),
          cell->level(),
          cell->index(),
          &dof_handler_dim);

        fe_eval.reinit(dof_cell);
        fe_eval_dim.reinit(dof_cell_dim);

        curvature_temp.reinit(fe_eval.dofs_per_cell);

        std::vector<std::vector<Tensor<1, spacedim, double>>> normal_gradients(
          fe_eval.dofs_per_cell, std::vector<Tensor<1, spacedim, double>>(spacedim));

        fe_eval_dim.get_function_gradients(normal_vector, normal_gradients);

        for (const auto q : fe_eval_dim.quadrature_point_indices())
          {
            double curvature = 0.0;

            for (unsigned c = 0; c < spacedim; ++c)
              curvature += normal_gradients[q][c][c];

            curvature_temp[q] = curvature;
          }

        dof_cell->set_dof_values(curvature_temp, curvature_vector);
      }
  }



  template <int dim, int spacedim>
  std::tuple<std::vector<std::pair<int, int>>,
             std::vector<unsigned int>,
             std::vector<double>,
             std::vector<Point<spacedim>>>
  collect_evaluation_points(const Triangulation<dim, spacedim> &     surface_mesh,
                            const Mapping<dim, spacedim> &           surface_mapping,
                            const FiniteElement<dim, spacedim> &     surface_fe,
                            const Quadrature<dim> &                  surface_quad,
                            const Triangulation<spacedim, spacedim> &tria,
                            const Mapping<spacedim, spacedim> &      mapping)
  {
    // step 1: determine quadrature points in real coordinate system and quadrature weight
    std::vector<std::pair<Point<spacedim>, double>> locally_owned_surface_points;

    FEValues<dim, spacedim> fe_eval(surface_mapping,
                                    surface_fe,
                                    surface_quad,
                                    update_quadrature_points | update_JxW_values);

    for (const auto &cell : surface_mesh.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        fe_eval.reinit(cell);

        for (const auto q : fe_eval.quadrature_point_indices())
          locally_owned_surface_points.emplace_back(fe_eval.quadrature_point(q),
                                                    fe_eval.JxW(q));
      }

    // step 2 communicate (TODO)

    // step 3: convert quadrature points to a pair of cells and reference cell quadrature
    // point
    std::vector<std::tuple<Point<spacedim>, double, std::pair<int, int>>> info;

    const std::vector<bool>                    marked_vertices;
    const GridTools::Cache<spacedim, spacedim> cache(tria, mapping);
    const double                               tolerance = 1e-10;
    auto                                       cell_hint = tria.begin_active();

    for (const auto &point_and_weight : locally_owned_surface_points)
      {
        const auto cell_and_reference_coordinate =
          GridTools::find_active_cell_around_point(
            cache, point_and_weight.first, cell_hint, marked_vertices, tolerance);

        cell_hint = cell_and_reference_coordinate.first;

        info.emplace_back(
          cell_and_reference_coordinate.second,
          point_and_weight.second,
          std::pair<int, int>(cell_and_reference_coordinate.first->level(),
                              cell_and_reference_coordinate.first->index()));
      }

    // step 4: compress data structures
    std::sort(info.begin(), info.end(), [](const auto &a, const auto &b) {
      return std::get<2>(a) < std::get<2>(b);
    });

    std::vector<std::pair<int, int>> cells;
    std::vector<unsigned int>        ptrs;
    std::vector<double>              weights;
    std::vector<Point<spacedim>>     points;

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



  template <int dim, typename VectorType, typename BlockVectorType>
  void
  compute_force_vector_sharp_interface(const Triangulation<dim - 1, dim> &surface_mesh,
                                       const Mapping<dim - 1, dim> &      surface_mapping,
                                       const FiniteElement<dim - 1, dim> &surface_fe,
                                       const Quadrature<dim - 1> &        surface_quad,
                                       const Mapping<dim> &               mapping,
                                       const DoFHandler<dim> &            dof_handler,
                                       const DoFHandler<dim> &            dof_handler_dim,
                                       const BlockVectorType &normal_vector_field,
                                       const VectorType &     curvature_solution,
                                       VectorType &           force_vector)
  {
    // step 1) collect all locally-relevant surface quadrature points (cell,
    // reference-cell position,
    //  quadrature weight)
    const auto [cells, ptrs, weights, points] =
      collect_evaluation_points(surface_mesh,
                                surface_mapping,
                                surface_fe,
                                surface_quad,
                                dof_handler.get_triangulation(),
                                mapping);

    // step 2) loop over all cells and evaluate curvature and normal in the cell-local
    // quadrature points
    //   and test with all test functions of the cell

    AffineConstraints<double> constraints; // TODO: use the right ones

    FEPointEvaluation<1, dim> phi_curvature(mapping, dof_handler.get_fe());

    FESystem<dim>               fe_dim(dof_handler.get_fe(), dim);
    FEPointEvaluation<dim, dim> phi_normal_force(mapping, fe_dim);

    std::vector<double>                  buffer;
    std::vector<double>                  buffer_dim;
    std::vector<types::global_dof_index> local_dof_indices;

    for (unsigned int i = 0; i < cells.size(); ++i)
      {
        typename DoFHandler<dim>::active_cell_iterator cell = {
          &dof_handler.get_triangulation(),
          cells[i].first,
          cells[i].second,
          &dof_handler};

        typename DoFHandler<dim>::active_cell_iterator cell_dim = {
          &dof_handler.get_triangulation(),
          cells[i].first,
          cells[i].second,
          &dof_handler_dim};

        const unsigned int n_dofs_per_component = cell->get_fe().n_dofs_per_cell();

        local_dof_indices.resize(n_dofs_per_component);
        buffer.resize(n_dofs_per_component);
        buffer_dim.resize(n_dofs_per_component * dim);

        cell->get_dof_indices(local_dof_indices);

        const unsigned int n_points = ptrs[i + 1] - ptrs[i];

        const ArrayView<const Point<dim>> unit_points(points.data() + ptrs[i], n_points);
        const ArrayView<const double>     JxW(weights.data() + ptrs[i], n_points);

        // gather curvature
        constraints.get_dof_values(curvature_solution,
                                   local_dof_indices.begin(),
                                   buffer.begin(),
                                   buffer.end());

        // evaluate curvature
        phi_curvature.evaluate(cell,
                               unit_points,
                               make_array_view(buffer),
                               EvaluationFlags::values);

        // gather normal
        for (int i = 0; i < dim; ++i)
          {
            constraints.get_dof_values(normal_vector_field.block(i),
                                       local_dof_indices.begin(),
                                       buffer.begin(),
                                       buffer.end());
            for (unsigned int c = 0; c < n_dofs_per_component; ++c)
              buffer_dim[c * dim + i] = buffer[c];
          }

        // evaluate normal
        phi_normal_force.evaluate(cell, unit_points, buffer_dim, EvaluationFlags::values);

        // quadrature loop
        for (unsigned int q = 0; q < n_points; ++q)
          {
            const auto normal =
              phi_normal_force.get_value(q) / phi_normal_force.get_value(q).norm();
            phi_normal_force.submit_value(normal * phi_curvature.get_value(q) * JxW[q],
                                          q);
          }

        // integrate force
        phi_normal_force.integrate(cell,
                                   unit_points,
                                   buffer_dim,
                                   EvaluationFlags::values);

        local_dof_indices.resize(n_dofs_per_component * dim);
        cell_dim->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(buffer_dim,
                                               local_dof_indices /*TODO*/,
                                               force_vector);
      }
  }



  class SharpInterfaceSolver
  {
  public:
    virtual void
    advance_time_step() = 0;

    virtual void
    output_solution(const std::string &output_filename) = 0;
  };



  template <int dim>
  class FrontTrackingSolver : public SharpInterfaceSolver
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    FrontTrackingSolver(NavierStokes<dim> &          navier_stokes_solver,
                        Triangulation<dim - 1, dim> &surface_mesh)
      : navier_stokes_solver(navier_stokes_solver)
      , euler_dofhandler(surface_mesh)
      , surface_dofhandler(surface_mesh)
    {
      const unsigned int fe_degree      = 3;
      const unsigned int mapping_degree = fe_degree;

      FESystem<dim - 1, dim> euler_fe(FE_Q<dim - 1, dim>(fe_degree), dim);
      euler_dofhandler.distribute_dofs(euler_fe);

      surface_dofhandler.distribute_dofs(FE_Q<dim - 1, dim>(fe_degree));

      euler_vector.reinit(euler_dofhandler.n_dofs());
      VectorTools::get_position_vector(euler_dofhandler,
                                       euler_vector,
                                       MappingQGeneric<dim - 1, dim>(mapping_degree));

      euler_mapping =
        std::make_shared<MappingFEField<dim - 1, dim, VectorType>>(euler_dofhandler,
                                                                   euler_vector);

      this->update_phases();
      this->update_gravity_force();
      this->update_surface_tension();
    }

    void
    advance_time_step() override
    {
      this->move_surface_mesh();
      this->update_phases();
      this->update_gravity_force();
      this->update_surface_tension();

      navier_stokes_solver.get_constraints_u().set_zero(
        navier_stokes_solver.user_rhs.block(0));
      navier_stokes_solver.advance_time_step();
    }

    void
    output_solution(const std::string &output_filename) override
    {
      // background mesh
      {
        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;

        DataOut<dim> data_out;
        data_out.set_flags(flags);

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vector_component_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);

        navier_stokes_solver.solution.update_ghost_values();

        data_out.add_data_vector(navier_stokes_solver.get_dof_handler_u(),
                                 navier_stokes_solver.solution.block(0),
                                 std::vector<std::string>(dim, "velocity"),
                                 vector_component_interpretation);

        data_out.add_data_vector(navier_stokes_solver.get_dof_handler_u(),
                                 navier_stokes_solver.user_rhs.block(0),
                                 std::vector<std::string>(dim, "user_rhs"),
                                 vector_component_interpretation);

        data_out.add_data_vector(navier_stokes_solver.get_dof_handler_p(),
                                 navier_stokes_solver.solution.block(1),
                                 "pressure");

        data_out.build_patches(navier_stokes_solver.mapping,
                               navier_stokes_solver.get_dof_handler_u().get_fe().degree +
                                 1);

        navier_stokes_solver.write_data_output(
          output_filename,
          navier_stokes_solver.time_stepping,
          navier_stokes_solver.get_parameters().output_frequency,
          navier_stokes_solver.get_dof_handler_u().get_triangulation(),
          data_out);
      }

      // surface mesh
      {
        DataOutBase::VtkFlags flags;

        DataOut<dim - 1, DoFHandler<dim - 1, dim>> data_out;
        data_out.set_flags(flags);
        data_out.add_data_vector(surface_dofhandler, curvature_vector, "curvature");
        data_out.add_data_vector(euler_dofhandler, normal_vector, "normal");

        data_out.build_patches(
          *euler_mapping,
          euler_dofhandler.get_fe().degree + 1,
          DataOut<dim - 1,
                  DoFHandler<dim - 1, dim>>::CurvedCellRegion::curved_inner_cells);

        std::filesystem::path path(output_filename + "_surface");

        data_out.write_vtu_with_pvtu_record(path.parent_path().string() + "/",
                                            path.filename(),
                                            navier_stokes_solver.time_stepping.step_no(),
                                            MPI_COMM_WORLD);
      }
    }

  private:
    void
    move_surface_mesh()
    {
      VectorTools::update_position_vector(navier_stokes_solver.time_stepping.step_size(),
                                          navier_stokes_solver.get_dof_handler_u(),
                                          navier_stokes_solver.mapping,
                                          navier_stokes_solver.solution.block(0),
                                          euler_dofhandler,
                                          *euler_mapping,
                                          euler_vector);
    }

    void
    update_phases()
    {
      const auto density        = navier_stokes_solver.get_parameters().density;
      const auto density_diff   = navier_stokes_solver.get_parameters().density_diff;
      const auto viscosity      = navier_stokes_solver.get_parameters().viscosity;
      const auto viscosity_diff = navier_stokes_solver.get_parameters().viscosity_diff;

      if (density_diff == 0.0 && viscosity_diff == 0.0)
        return; // nothing to do

      boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double>>
        polygon;
      GridTools::construct_polygon(*euler_mapping, euler_dofhandler, polygon);

      double dummy;

      navier_stokes_solver.matrix_free->template cell_loop<double, double>(
        [&](const auto &matrix_free, auto &, const auto &, auto macro_cells) {
          FEEvaluation<dim, -1, 0, 1, double> phi(matrix_free, 0, 0);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              phi.reinit(cell);

              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  const auto indicator =
                    GridTools::within(polygon, phi.quadrature_point(q));

                  navier_stokes_solver.get_matrix().begin_densities(cell)[q] =
                    density + density_diff * indicator;
                  navier_stokes_solver.get_matrix().begin_viscosities(cell)[q] =
                    viscosity + viscosity_diff * indicator;
                }
            }
        },
        dummy,
        dummy);
    }

    void
    update_surface_tension()
    {
      // return; // TODO: not working

      normal_vector.reinit(euler_dofhandler.n_dofs());
      curvature_vector.reinit(surface_dofhandler.n_dofs());

      compute_normal(*euler_mapping, euler_dofhandler, normal_vector);
      compute_curvature(*euler_mapping,
                        euler_dofhandler,
                        surface_dofhandler,
                        QGaussLobatto<dim - 1>(surface_dofhandler.get_fe().degree + 1),
                        normal_vector,
                        curvature_vector);

      compute_force_vector_sharp_interface(
        *euler_mapping,
        surface_dofhandler,
        euler_dofhandler,
        QGauss<dim - 1>(euler_dofhandler.get_fe().degree + 1),
        navier_stokes_solver.mapping,
        navier_stokes_solver.get_dof_handler_u(),
        navier_stokes_solver.get_parameters().surface_tension,
        normal_vector,
        curvature_vector,
        navier_stokes_solver.user_rhs.block(0));
    }

    void
    update_gravity_force()
    {
      const auto gravity = navier_stokes_solver.get_parameters().gravity;

      const auto density      = navier_stokes_solver.get_parameters().density;
      const auto density_diff = navier_stokes_solver.get_parameters().density_diff;

      const bool zero_out = true;

      navier_stokes_solver.matrix_free->template cell_loop<VectorType, std::nullptr_t>(
        [&](const auto &matrix_free, auto &vec, const auto &, auto macro_cells) {
          FEEvaluation<dim, -1, 0, dim, double> phi(matrix_free, 0, 0);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              phi.reinit(cell);

              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  Tensor<1, dim, VectorizedArray<double>> force;

                  force[dim - 1] -=
                    gravity *
                    (density_diff == 0.0 ?
                       VectorizedArray<double>(density) :
                       navier_stokes_solver.get_matrix().begin_densities(cell)[q]);
                  phi.submit_value(force, q);
                }
              phi.integrate_scatter(true, false, vec);
            }
        },
        navier_stokes_solver.user_rhs.block(0),
        nullptr,
        zero_out);
    }

    // background mesh
    NavierStokes<dim> &navier_stokes_solver;

    // surface mesh
    DoFHandler<dim - 1, dim>               euler_dofhandler;
    DoFHandler<dim - 1, dim>               surface_dofhandler;
    VectorType                             euler_vector;
    std::shared_ptr<Mapping<dim - 1, dim>> euler_mapping;

    VectorType normal_vector;
    VectorType curvature_vector;
  };



  template <int dim>
  class LevelSetSolver
  {
  public:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    static const unsigned int dof_index_ls        = 0;
    static const unsigned int dof_index_normal    = 1;
    static const unsigned int dof_index_curvature = 2;
    static const unsigned int quad_index          = 0;

    LevelSetSolver(
      const FlowParameters &parameters,
      const TimeStepping &  time_stepping,
      VectorType &          velocity_solution,
      VectorType &          velocity_solution_old,
      VectorType &          velocity_solution_old_old,
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &fluid_type,
      const std::set<types::boundary_id> &                                symmetry)
      : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , parameters(parameters)
      , time_stepping(time_stepping)
      , last_concentration_range(-1, +1)
      , first_reinit_step(true)
      , velocity_solution(velocity_solution)
      , velocity_solution_old(velocity_solution_old)
      , velocity_solution_old_old(velocity_solution_old_old)
      , normal_vector_field(dim)
      , normal_vector_rhs(dim)
    {
      {
        LevelSetOKZSolverComputeNormalParameter params;
        params.dof_index_ls            = dof_index_ls;
        params.dof_index_normal        = dof_index_normal;
        params.quad_index              = quad_index;
        params.epsilon                 = this->parameters.epsilon;
        params.approximate_projections = this->parameters.approximate_projections;

        normal_operator =
          std::make_unique<LevelSetOKZSolverComputeNormal<dim>>(normal_vector_field,
                                                                normal_vector_rhs,
                                                                ls_solution,
                                                                cell_diameters,
                                                                epsilon_used,
                                                                minimal_edge_length,
                                                                constraints_normals,
                                                                params,
                                                                matrix_free,
                                                                preconditioner,
                                                                projection_matrix,
                                                                ilu_projection_matrix);
      }

      {
        LevelSetOKZSolverReinitializationParameter params;
        params.dof_index_ls     = dof_index_ls;
        params.dof_index_normal = dof_index_normal;
        params.quad_index       = quad_index;
        params.do_iteration     = this->parameters.do_iteration;

        params.time.time_step_scheme     = this->parameters.time_step_scheme;
        params.time.start_time           = this->parameters.start_time;
        params.time.end_time             = this->parameters.end_time;
        params.time.time_step_size_start = this->parameters.time_step_size_start;
        params.time.time_stepping_cfl    = this->parameters.time_stepping_cfl;
        params.time.time_stepping_coef2  = this->parameters.time_stepping_coef2;
        params.time.time_step_tolerance  = this->parameters.time_step_tolerance;
        params.time.time_step_size_max   = this->parameters.time_step_size_max;
        params.time.time_step_size_min   = this->parameters.time_step_size_min;

        reinit = std::make_unique<LevelSetOKZSolverReinitialization<dim>>(
          normal_vector_field,
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
          params,
          first_reinit_step,
          matrix_free);
      }

      {
        LevelSetOKZSolverComputeCurvatureParameter params;
        params.dof_index_ls            = dof_index_ls;
        params.dof_index_curvature     = dof_index_curvature;
        params.dof_index_normal        = dof_index_normal;
        params.quad_index              = quad_index;
        params.epsilon                 = this->parameters.epsilon;
        params.approximate_projections = this->parameters.approximate_projections;
        params.curvature_correction    = this->parameters.curvature_correction;

        curvature_operator =
          std::make_unique<LevelSetOKZSolverComputeCurvature<dim>>(cell_diameters,
                                                                   normal_vector_field,
                                                                   constraints_curvature,
                                                                   constraints,
                                                                   epsilon_used,
                                                                   curvature_rhs,
                                                                   params,
                                                                   curvature_solution,
                                                                   ls_solution,
                                                                   matrix_free,
                                                                   preconditioner,
                                                                   projection_matrix,
                                                                   ilu_projection_matrix);
      }

      {
        LevelSetOKZSolverAdvanceConcentrationParameter params;

        params.dof_index_ls             = 2;
        params.dof_index_vel            = 0;
        params.quad_index               = 2;
        params.convection_stabilization = this->parameters.convection_stabilization;
        params.do_iteration             = this->parameters.do_iteration;
        params.tol_nl_iteration         = this->parameters.tol_nl_iteration;

        LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor<dim> bcs;

        bcs.dirichlet = fluid_type;
        bcs.symmetry  = symmetry;

        params.time.time_step_scheme     = this->parameters.time_step_scheme;
        params.time.start_time           = this->parameters.start_time;
        params.time.end_time             = this->parameters.end_time;
        params.time.time_step_size_start = this->parameters.time_step_size_start;
        params.time.time_stepping_cfl    = this->parameters.time_stepping_cfl;
        params.time.time_stepping_coef2  = this->parameters.time_stepping_coef2;
        params.time.time_step_tolerance  = this->parameters.time_step_tolerance;
        params.time.time_step_size_max   = this->parameters.time_step_size_max;
        params.time.time_step_size_min   = this->parameters.time_step_size_min;

        this->advection_operator =
          std::make_unique<LevelSetOKZSolverAdvanceConcentration<dim>>(
            ls_solution,
            ls_solution_old,
            ls_solution_old_old,
            ls_update,
            ls_rhs,
            velocity_solution,
            velocity_solution_old,
            velocity_solution_old_old,
            cell_diameters,
            this->constraints,
            this->pcout,
            bcs,
            this->matrix_free,
            params,
            this->preconditioner);
      }

      // MatrixFree
      {

      }

      // Vectors
      {
        matrix_free.initialize_dof_vector(ls_solution_update, dof_index_ls);
        matrix_free.initialize_dof_vector(ls_system_rhs, dof_index_ls);
        matrix_free.initialize_dof_vector(curvature_rhs, dof_index_curvature);

        for (unsigned int i = 0; i < dim; ++i)
          matrix_free.initialize_dof_vector(normal_vector_rhs.block(i), dof_index_normal);
      }

      // miscellaneous
      {
        compute_cell_diameters(
          matrix_free, dof_index_ls, cell_diameters, minimal_edge_length, epsilon_used);

        epsilon_used =
          parameters.epsilon / parameters.concentration_subdivisions * epsilon_used;

        initialize_mass_matrix_diagonal(matrix_free,
                                        hanging_node_constraints,
                                        dof_index_ls,
                                        quad_index,
                                        preconditioner);

        initialize_projection_matrix(matrix_free,
                                     constraints_normals,
                                     dof_index_ls,
                                     quad_index,
                                     epsilon_used,
                                     this->parameters.epsilon,
                                     cell_diameters,
                                     *projection_matrix,
                                     *ilu_projection_matrix);
      }
    }

    void
    solve()
    {
      this->advance_concentration();
      this->reinitialize();
    }

    const VectorType &
    get_level_set_vector()
    {
      return ls_solution;
    }

    const BlockVectorType &
    get_normal_vector()
    {
      return normal_vector_field;
    }

    const VectorType &
    get_curvature_vector()
    {
      return curvature_solution;
    }

    const DoFHandler<dim> &
    get_dof_handler() const
    {
      return dof_handler;
    }

  private:
    void
    advance_concentration()
    {
      advection_operator->advance_concentration(this->time_stepping.step_size());
    }

    void
    reinitialize()
    {
      const double       dt         = this->time_stepping.step_size();
      const unsigned int stab_steps = this->parameters.n_reinit_steps;
      const unsigned int diff_steps = 0;

      reinit->reinitialize(dt, stab_steps, diff_steps, [this](const bool fast) {
        normal_operator->compute_normal(fast);
      });
    }

    ConditionalOStream    pcout;
    const FlowParameters &parameters;
    const TimeStepping &  time_stepping;


    std::pair<double, double>              last_concentration_range;
    bool                                   first_reinit_step;
    AlignedVector<VectorizedArray<double>> cell_diameters;
    double                                 minimal_edge_length;
    double                                 epsilon_used;

    DoFHandler<dim> dof_handler;

    MatrixFree<dim, double>   matrix_free;
    AffineConstraints<double> constraints;
    AffineConstraints<double> constraints_normals;
    AffineConstraints<double> hanging_node_constraints;
    AffineConstraints<double> constraints_curvature;

    VectorType &velocity_solution;
    VectorType &velocity_solution_old;
    VectorType &velocity_solution_old_old;

    VectorType      ls_solution, ls_solution_old, ls_solution_old_old, ls_update, ls_rhs;
    BlockVectorType normal_vector_field;
    VectorType      curvature_solution;

    BlockVectorType normal_vector_rhs;
    VectorType      ls_solution_update;
    VectorType      ls_system_rhs;
    VectorType      curvature_rhs;

    DiagonalPreconditioner<double>        preconditioner;
    std::shared_ptr<BlockMatrixExtension> projection_matrix;
    std::shared_ptr<BlockILUExtension>    ilu_projection_matrix;

    std::unique_ptr<LevelSetOKZSolverComputeNormal<dim>>        normal_operator;
    std::unique_ptr<LevelSetOKZSolverReinitialization<dim>>     reinit;
    std::unique_ptr<LevelSetOKZSolverComputeCurvature<dim>>     curvature_operator;
    std::unique_ptr<LevelSetOKZSolverAdvanceConcentration<dim>> advection_operator;
  };



  template <int dim>
  class MixedLevelSetSolver : public SharpInterfaceSolver
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    MixedLevelSetSolver(NavierStokes<dim> &          navier_stokes_solver,
                        Triangulation<dim - 1, dim> &surface_mesh,
                        const Function<dim> &        initial_values_ls)
      : navier_stokes_solver(navier_stokes_solver)
      , level_set_solver(navier_stokes_solver.get_parameters(),
                         navier_stokes_solver.time_stepping,
                         navier_stokes_solver.solution.block(0),
                         navier_stokes_solver.solution_old.block(0),
                         navier_stokes_solver.solution_old_old.block(0),
                         navier_stokes_solver.boundary->fluid_type,
                         navier_stokes_solver.boundary->symmetry)
      , euler_dofhandler(surface_mesh)
      , surface_dofhandler(surface_mesh)
    {
      (void)initial_values_ls;
    }

    void
    advance_time_step() override
    {
      level_set_solver.solve();

      this->move_surface_mesh();
      this->update_phases();
      this->update_gravity_force();
      this->update_surface_tension();

      navier_stokes_solver.get_constraints_u().set_zero(
        navier_stokes_solver.user_rhs.block(0));
      navier_stokes_solver.advance_time_step();
    }

    void
    output_solution(const std::string &output_filename) override
    {
      // background mesh
      {
        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;

        DataOut<dim> data_out;
        data_out.set_flags(flags);

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vector_component_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);

        navier_stokes_solver.solution.update_ghost_values();

        data_out.add_data_vector(navier_stokes_solver.get_dof_handler_u(),
                                 navier_stokes_solver.solution.block(0),
                                 std::vector<std::string>(dim, "velocity"),
                                 vector_component_interpretation);

        data_out.add_data_vector(navier_stokes_solver.get_dof_handler_u(),
                                 navier_stokes_solver.user_rhs.block(0),
                                 std::vector<std::string>(dim, "user_rhs"),
                                 vector_component_interpretation);

        data_out.add_data_vector(navier_stokes_solver.get_dof_handler_p(),
                                 navier_stokes_solver.solution.block(1),
                                 "pressure");

        data_out.add_data_vector(level_set_solver.get_dof_handler(),
                                 level_set_solver.get_curvature_vector(),
                                 "curvature");

        for (unsigned int i = 0; i < dim; ++i)
          data_out.add_data_vector(level_set_solver.get_dof_handler(),
                                   level_set_solver.get_normal_vector().block(i),
                                   "normal_" + std::to_string(i));

        data_out.build_patches(navier_stokes_solver.mapping,
                               navier_stokes_solver.get_dof_handler_u().get_fe().degree +
                                 1);

        navier_stokes_solver.write_data_output(
          output_filename,
          navier_stokes_solver.time_stepping,
          navier_stokes_solver.get_parameters().output_frequency,
          navier_stokes_solver.get_dof_handler_u().get_triangulation(),
          data_out);
      }

      // surface mesh
      {
        DataOutBase::VtkFlags flags;

        DataOut<dim - 1, DoFHandler<dim - 1, dim>> data_out;
        data_out.set_flags(flags);

        data_out.build_patches(
          *euler_mapping,
          euler_dofhandler.get_fe().degree + 1,
          DataOut<dim - 1,
                  DoFHandler<dim - 1, dim>>::CurvedCellRegion::curved_inner_cells);

        std::filesystem::path path(output_filename + "_surface");

        data_out.write_vtu_with_pvtu_record(path.parent_path().string() + "/",
                                            path.filename(),
                                            navier_stokes_solver.time_stepping.step_no(),
                                            MPI_COMM_WORLD);
      }
    }

  private:
    void
    move_surface_mesh()
    {
      VectorTools::update_position_vector(navier_stokes_solver.time_stepping.step_size(),
                                          navier_stokes_solver.get_dof_handler_u(),
                                          navier_stokes_solver.mapping,
                                          navier_stokes_solver.solution.block(0),
                                          euler_dofhandler,
                                          *euler_mapping,
                                          euler_vector);
    }

    void
    update_phases()
    {
      const auto density        = navier_stokes_solver.get_parameters().density;
      const auto density_diff   = navier_stokes_solver.get_parameters().density_diff;
      const auto viscosity      = navier_stokes_solver.get_parameters().viscosity;
      const auto viscosity_diff = navier_stokes_solver.get_parameters().viscosity_diff;

      if (density_diff == 0.0 && viscosity_diff == 0.0)
        return; // nothing to do

      double dummy;

      // TODO: select proper MatrixFree object and set right dof/quad index
      navier_stokes_solver.matrix_free->template cell_loop<double, VectorType>(
        [&](const auto &matrix_free, auto &, const auto &src, auto macro_cells) {
          FEEvaluation<dim, -1, 0, 1, double> phi(matrix_free, 0, 0);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              phi.reinit(cell);
              phi.gather_evaluate(src, EvaluationFlags::values);

              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  const auto indicator =
                    (phi.get_value(q) + 1.0) / 2.0; // TODO: fix indicator -> Heaviside

                  navier_stokes_solver.get_matrix().begin_densities(cell)[q] =
                    density + density_diff * indicator;
                  navier_stokes_solver.get_matrix().begin_viscosities(cell)[q] =
                    viscosity + viscosity_diff * indicator;
                }
            }
        },
        dummy,
        level_set_solver.get_level_set_vector());
    }

    void
    update_surface_tension()
    {
      compute_force_vector_sharp_interface<dim>(surface_dofhandler.get_triangulation(),
                                                *euler_mapping,
                                                surface_dofhandler.get_fe(),
                                                QGauss<dim - 1>(
                                                  surface_dofhandler.get_fe().degree + 1),
                                                navier_stokes_solver.mapping,
                                                level_set_solver.get_dof_handler(),
                                                navier_stokes_solver.get_dof_handler_u(),
                                                level_set_solver.get_normal_vector(),
                                                level_set_solver.get_curvature_vector(),
                                                navier_stokes_solver.user_rhs.block(0));
    }

    void
    update_gravity_force()
    {
      const auto gravity = navier_stokes_solver.get_parameters().gravity;

      const auto density      = navier_stokes_solver.get_parameters().density;
      const auto density_diff = navier_stokes_solver.get_parameters().density_diff;

      const bool zero_out = true;

      navier_stokes_solver.matrix_free->template cell_loop<VectorType, std::nullptr_t>(
        [&](const auto &matrix_free, auto &vec, const auto &, auto macro_cells) {
          FEEvaluation<dim, -1, 0, dim, double> phi(matrix_free, 0, 0);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              phi.reinit(cell);

              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  Tensor<1, dim, VectorizedArray<double>> force;

                  force[dim - 1] -=
                    gravity *
                    (density_diff == 0.0 ?
                       VectorizedArray<double>(density) :
                       navier_stokes_solver.get_matrix().begin_densities(cell)[q]);
                  phi.submit_value(force, q);
                }
              phi.integrate_scatter(true, false, vec);
            }
        },
        navier_stokes_solver.user_rhs.block(0),
        nullptr,
        zero_out);
    }

    // background mesh
    NavierStokes<dim> & navier_stokes_solver;
    LevelSetSolver<dim> level_set_solver;

    // surface mesh
    DoFHandler<dim - 1, dim>               euler_dofhandler;
    DoFHandler<dim - 1, dim>               surface_dofhandler;
    VectorType                             euler_vector;
    std::shared_ptr<Mapping<dim - 1, dim>> euler_mapping;
  };

} // namespace dealii
