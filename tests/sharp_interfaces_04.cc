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

// runs a simulation on a static bubble where the velocities ideally should be
// zero but where we actually get some velocities which are due to
// inaccuracies in the scheme

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <adaflo/level_set_okz.h>
#include <adaflo/level_set_okz_matrix.h>
#include <adaflo/parameters.h>
#include <adaflo/phase_field.h>

#include "sharp_interfaces_util.h"


using namespace dealii;

struct TwoPhaseParameters : public FlowParameters
{
  TwoPhaseParameters(const std::string &parameter_filename)
  {
    ParameterHandler prm;
    FlowParameters::declare_parameters(prm);
    prm.enter_subsection("Problem-specific");
    prm.declare_entry("two-phase method",
                      "front tracking",
                      Patterns::Selection(
                        "front tracking|mixed level set|sharp level set|level set"),
                      "Defines the two-phase method to be used");
    prm.leave_subsection();
    check_for_file(parameter_filename, prm);
    parse_parameters(parameter_filename, prm);
    prm.enter_subsection("Problem-specific");
    solver_method = prm.get("two-phase method");
    prm.leave_subsection();
  }

  std::string solver_method;
};



template <int dim>
class InitialValuesLS : public Function<dim>
{
public:
  InitialValuesLS()
    : Function<dim>(1, 0)
  {}

  double
  value(const Point<dim> &p, const unsigned int /*component*/) const
  {
    const double radius               = 0.25;
    Point<dim>   distance_from_origin = p;
    for (unsigned int i = 0; i < dim; ++i)
      distance_from_origin[i] = 0.5;
    return p.distance(distance_from_origin) - radius;
  }
};



template <int dim>
class MicroFluidicProblem
{
public:
  MicroFluidicProblem(const TwoPhaseParameters &parameters);
                       
  void
  run();

private:
void
  evaluate_spurious_velocities(NavierStokes<dim> &navier_stokes_solver);
/*
  std::pair<double, double>
  get_concentration_range(
    LevelSetSolver<2>          &level_set_solver) const;

  std::vector<double> compute_bubble_statistics(
    LevelSetSolver<2>          &level_set_solver,
    NavierStokes<2>            &navier_stokes_solver,
    std::vector<Tensor<2, dim>> *interface_points = 0,
    const unsigned int           sub_refinements  = numbers::invalid_unsigned_int) const;
*/
  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;

  mutable TimerOutput timer;

  TwoPhaseParameters                        parameters;
  parallel::distributed::Triangulation<dim> triangulation;

  std::unique_ptr<TwoPhaseBaseAlgorithm<dim>> two_phase_solver;
  //NavierStokes<dim> navier_stokes_solver(parameters, triangulation, &timer);
  
  std::vector<std::vector<double>> solution_data_spc;
  std::vector<std::vector<double>> solution_data;
  double global_omega_diameter;
  std::pair<double, double> concentration;
  mutable std::pair<double, double>      last_concentration_range;
};

template <int dim>
MicroFluidicProblem<dim>::MicroFluidicProblem(const TwoPhaseParameters &parameters)
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , timer(pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  , parameters(parameters)
  , triangulation(mpi_communicator)
  /*, navier_stokes_solver(NavierStokes<dim> &navier_stokes_solver)
  , level_set_solver(navier_stokes_solver.get_dof_handler_u().get_triangulation(),
                       InitialValuesLS<dim>(),
                       navier_stokes_solver.get_parameters(),
                       navier_stokes_solver.time_stepping,
                       navier_stokes_solver.solution.block(0),
                       navier_stokes_solver.solution_old.block(0),
                       navier_stokes_solver.solution_old_old.block(0),
                       navier_stokes_solver.boundary->fluid_type,
                       navier_stokes_solver.boundary->symmetry)
                       */
{
  two_phase_solver = std::make_unique<LevelSetOKZSolver<dim>>(parameters, triangulation);
}

template <int dim>
void
MicroFluidicProblem<dim>::evaluate_spurious_velocities(NavierStokes<dim> &navier_stokes_solver)
{
  double               local_norm_velocity, norm_velocity;
  const QIterated<dim> quadrature_formula(QTrapez<1>(), parameters.velocity_degree + 2);
  const unsigned int   n_q_points = quadrature_formula.size();

  const MPI_Comm &         mpi_communicator = triangulation.get_communicator();
  FEValues<dim> fe_values(navier_stokes_solver.get_fe_u(), quadrature_formula, update_values);
  std::vector<Tensor<1, dim>> velocity_values(n_q_points);
  local_norm_velocity = 0;  

  const FEValuesExtractors::Vector velocities(0);

  typename DoFHandler<dim>::active_cell_iterator
    cell = navier_stokes_solver.get_dof_handler_u().begin_active(),
    endc = navier_stokes_solver.get_dof_handler_u().end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values[velocities].get_function_values(navier_stokes_solver.solution.block(0),
                                                  velocity_values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          local_norm_velocity = std::max(local_norm_velocity, velocity_values[q].norm());
      }
  norm_velocity = Utilities::MPI::max(local_norm_velocity, mpi_communicator);

  double pressure_jump = 0;
  {
    QGauss<dim>       quadrature_formula(parameters.velocity_degree + 1);
    QGauss<dim - 1>   face_quadrature_formula(parameters.velocity_degree + 1);
    FEValues<dim>     ns_values(navier_stokes_solver.get_fe_p(),
                            quadrature_formula,
                            update_values | update_JxW_values);
    FEFaceValues<dim> fe_face_values(navier_stokes_solver.get_fe_p(),
                                     face_quadrature_formula,
                                     update_values | update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    std::vector<double> p_values(n_q_points);
    std::vector<double> p_face_values(face_quadrature_formula.size());
          
    // With all this in place, we can go on with the loop over all cells and
    // add the local contributions.
    //
    // The first thing to do is to evaluate the FE basis functions at the
    // quadrature points of the cell, as well as derivatives and the other
    // quantities specified above.  Moreover, we need to reset the local
    // matrices and right hand side before filling them with new information
    // from the current cell.
    const FEValuesExtractors::Scalar p(dim);
    double pressure_average = 0, one_average = 0, press_b = 0, one_b = 0;
    typename DoFHandler<dim>::active_cell_iterator
      endc    = navier_stokes_solver.get_dof_handler_p().end(),
      ns_cell = navier_stokes_solver.get_dof_handler_p().begin_active();

    for (; ns_cell != endc; ++ns_cell)
      if (ns_cell->is_locally_owned())
        {
          ns_values.reinit(ns_cell);

          if (ns_cell->center().norm() < 0.1)
            {
              ns_values.get_function_values(navier_stokes_solver.solution.block(1), p_values);
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  pressure_average += p_values[q] * ns_values.JxW(q);
                  one_average += ns_values.JxW(q);
                }
               
            }
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
            if (ns_cell->face(face)->at_boundary())
              {
                fe_face_values.reinit(ns_cell, face);
                fe_face_values.get_function_values(navier_stokes_solver.solution.block(1),
                                                   p_face_values);
                for (unsigned int q = 0; q < face_quadrature_formula.size(); ++q)
                  {
                    press_b += p_face_values[q] * fe_face_values.JxW(q);
                    one_b += fe_face_values.JxW(q);
                  }
              }
        }

    const double global_p_avg = Utilities::MPI::sum(pressure_average, mpi_communicator);
    const double global_o_avg = Utilities::MPI::sum(one_average, mpi_communicator);
    const double global_p_bou = Utilities::MPI::sum(press_b, mpi_communicator);
    const double global_o_bou = Utilities::MPI::sum(one_b, mpi_communicator);
    pressure_jump = ((global_p_avg / global_o_avg - global_p_bou / global_o_bou) -
                     2. * (dim - 1) * parameters.surface_tension) /
                    (2 * (dim - 1) * parameters.surface_tension) * 100.;
    std::cout.precision(8);
    //pcout << "  pressure_average:  " << pressure_average << "   one_average: " << one_average << std::endl;
    //pcout << "  press_b:  " << press_b << "  one_b: " << one_b << std::endl;
    pcout << "  Error in pressure jump: " << pressure_jump << " %" << std::endl;
  }

  // calculate spurious currents
  pcout << "  Size spurious currents, absolute: " << norm_velocity << std::endl;

  // TODO: Do I need this?
  std::vector<double> data(3);
  data[0] = navier_stokes_solver.time_stepping.now();
  data[1] = norm_velocity;
  data[2] = pressure_jump;
  if (solution_data_spc.size() && data[0] == solution_data_spc.back()[0])
    solution_data_spc.back().insert(solution_data.back().end(), data.begin() + 1, data.end());
  else
    solution_data_spc.push_back(data);
}

/*
template <>
std::pair<double, double>
MicroFluidicProblem<2>::get_concentration_range(
  LevelSetSolver<2>         &level_set_solver) const
{
  const unsigned int dim = 2;
  const QIterated<dim> quadrature_formula(QTrapez<1>(), level_set_solver.get_fe_ls().degree + 2);
  FEValues<dim>        fe_values(level_set_solver.get_fe_ls(), quadrature_formula, update_values);
  const unsigned int   n_q_points = quadrature_formula.size();
  std::vector<double>  concentration_values(n_q_points);
  Vector<double>          sol_values(level_set_solver.get_fe_ls().dofs_per_cell);

  double min_concentration = std::numeric_limits<double>::max(),
         max_concentration = -min_concentration;

  typename DoFHandler<dim>::active_cell_iterator cell = level_set_solver.get_dof_handler().begin_active(),
                                                 endc = level_set_solver.get_dof_handler().end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(level_set_solver.get_level_set_vector(), concentration_values);
        //TODO: remove, just for test cases
        /* cell->get_interpolated_dof_values(level_set_solver.get_level_set_vector(), sol_values);
        //fe_values.get_function_values(level_set_solver.get_level_set_vector(), ls_values);
        for (unsigned int i = 1; i < level_set_solver.get_fe_ls().dofs_per_cell; ++i){
            if (sol_values(i) * sol_values(0) <= 0){
              pcout <<  " i = " << i << "   ls_values = " << sol_values(i) << std::endl;
            }
        }*/
 /*       for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double concentration = concentration_values[q];
            //pcout << "q = " << q << " : concentration = " << concentration << std::endl;

            min_concentration = std::min(min_concentration, concentration);
            max_concentration = std::max(max_concentration, concentration);
          }
       }
  last_concentration_range = std::make_pair(
    -Utilities::MPI::max(-min_concentration, get_communicator(triangulation)),
    Utilities::MPI::max(max_concentration, get_communicator(triangulation)));
  return last_concentration_range;
}
*/

/*
template <>
std::vector<double> MicroFluidicProblem<2>::compute_bubble_statistics(
  LevelSetSolver<2>         &level_set_solver,
  NavierStokes<2>           &navier_stokes_solver,
  std::vector<Tensor<2, 2>> *interface_points,
  const unsigned int         sub_refinements) const
{
  pcout << "compute bubble statistics dimension 2" << std::endl;

  const unsigned int dim = 2;

  const int sub_per_d = 1; /*sub_refinements == numbers::invalid_unsigned_int ?
                          parameters.velocity_degree + 3 :
                          sub_refinements;*/
/*  const QIterated<dim> quadrature_formula(QTrapez<1>(), sub_per_d);
  const QGauss<dim>    interior_quadrature(parameters.velocity_degree);
  const unsigned int   n_q_points = quadrature_formula.size();
  FEValues<dim>        fe_values(navier_stokes_solver.mapping,
                          level_set_solver.get_dof_handler().get_fe(),
                          quadrature_formula,
                          update_values | update_JxW_values | update_quadrature_points);
  FEValues<dim>        ns_values(navier_stokes_solver.mapping,
                          navier_stokes_solver.get_fe_u(),
                          quadrature_formula,
                          update_values);
  FEValues<dim>        interior_ns_values(navier_stokes_solver.mapping,
                                   navier_stokes_solver.get_fe_u(),
                                   interior_quadrature,
                                   update_values | update_JxW_values |
                                     update_quadrature_points);

  const FEValuesExtractors::Vector vel(0);

  const unsigned int n_points       = 2 * (dim > 1 ? 2 : 1) * (dim > 2 ? 2 : 1),
                     n_subdivisions = (sub_per_d) * (dim > 1 ? (sub_per_d) : 1) *
                                      (dim > 2 ? (sub_per_d) : 1);
  std::vector<double> full_c_values(n_q_points), c_values(n_points),
    quad_weights(n_points), weight_correction(n_q_points);
  std::vector<Tensor<1, dim>> velocity_values(n_q_points), velocities(n_points),
    int_velocity_values(interior_quadrature.size());
  std::vector<Point<dim>> quad(n_points);
  Vector<double>          sol_values(level_set_solver.get_dof_handler().get_fe().dofs_per_cell);
  std::vector<double>          ls_values(n_q_points);


  for (unsigned int i = 0; i < n_q_points; i++)
    {
      weight_correction[i] = 1;
      unsigned int fact    = sub_per_d + 1;
      if (i % fact > 0 && i % fact < fact - 1)
        weight_correction[i] *= 0.5;
      if (i >= fact && i < n_q_points - fact)
        weight_correction[i] *= 0.5;
    }

  if (interface_points != 0)
    interface_points->clear();
  double                                area = 0, perimeter = 0;
  Tensor<1, dim>                        center_of_mass, velocity;

  level_set_solver.get_level_set_vector().update_ghost_values();
  
  DoFHandler<dim>::active_cell_iterator cell = level_set_solver.get_dof_handler().begin_active(),
                                        endc = level_set_solver.get_dof_handler().end();
  
  DoFHandler<dim>::active_cell_iterator ns_cell =
    navier_stokes_solver.get_dof_handler_u().begin_active();
  for (; cell != endc; ++cell, ++ns_cell)
    if (cell->is_locally_owned())
      {
        // cheap test: find out whether the interface crosses this cell,
        // i.e. two solution values have a different sign. if not, can compute
        // with a low order Gauss quadrature without caring about the interface
        //TODO: right? ls value not velocity or so?
       //fe_values.reinit(cell);
        //fe_values.get_function_values(level_set_solver.get_level_set_vector(), ls_values);
        cell->get_interpolated_dof_values(level_set_solver.get_level_set_vector(), sol_values);
          //solution.block(0), sol_values);
        //pcout << "dofs fe ls = " << level_set_solver.get_fe_ls().dofs_per_cell << std::endl;
        bool interface_crosses_cell = false;
        for (unsigned int i = 1; i < level_set_solver.get_fe_ls().dofs_per_cell; ++i){
          if (sol_values(i) * sol_values(0) <= 0){
            interface_crosses_cell = true;
            pcout << "bs: i = " << i <<":   sol Value =  " << sol_values(i) << std::endl;
          }
        }
        if (interface_crosses_cell == false)
          {
            bool has_area = sol_values(0) > 0;
            interior_ns_values.reinit(ns_cell);
            interior_ns_values[vel].get_function_values(navier_stokes_solver.solution.block(0),
                                                        int_velocity_values);
            for (unsigned int q = 0; q < interior_quadrature.size(); q++)
              {
                if (has_area)
                  {
                    area += interior_ns_values.JxW(q);
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        center_of_mass[d] += (interior_ns_values.quadrature_point(q)[d] *
                                              interior_ns_values.JxW(q));
                        velocity[d] +=
                          (int_velocity_values[q][d] * interior_ns_values.JxW(q));
                      }
                  }
              }
            continue;
          }

        // when the interface crosses this cell, have to find the crossing
        // points (linear interpolation) and compute the area fraction
        fe_values.reinit(cell);
        // TODO: check if ls value right!
        //fe_values.get_function_values(solution.block(0), full_c_values);
        fe_values.get_function_values(level_set_solver.get_level_set_vector(), full_c_values);
        ns_values.reinit(ns_cell);
        ns_values[vel].get_function_values(navier_stokes_solver.solution.block(0),
                                           velocity_values);

        for (unsigned int d = 0; d < n_subdivisions; d++)
          {
            // compute a patch of four points
            {
              const int initial_shift = d % sub_per_d + (d / sub_per_d) * (sub_per_d + 1);
              for (unsigned int i = 0; i < n_points; i++)
                {
                  const unsigned int index =
                    initial_shift + (i / 2) * (sub_per_d - 1) + i;
                  Assert(index < n_q_points, ExcInternalError());
                  c_values[i]     = full_c_values[index];
                  velocities[i]   = velocity_values[index];
                  quad[i]         = fe_values.quadrature_point(index);
                  quad_weights[i] = fe_values.JxW(index) * weight_correction[index];
                  //pcout << "d = " << d << " i = " << i << "   full_c_values = " << full_c_values[index] << std::endl;
                }
            }
            double         local_area = 1;
            double         int_rx0 = -1, int_rx1 = -1, int_ry0 = -1, int_ry1 = -1;
            Tensor<1, dim> pos_x0, pos_x1, pos_y0, pos_y1;

            // add a small perturbation to avoid having exact zero values
            for (unsigned int i = 0; i < n_points; ++i)
              c_values[i] += 1e-22;

            // locate interface
            if (c_values[0] * c_values[1] <= 0)
              {
                int_rx0 = c_values[0] / (c_values[0] - c_values[1]);
                pos_x0  = quad[0] + (quad[1] - quad[0]) * int_rx0;
              }
            if (c_values[2] * c_values[3] <= 0)
              {
                int_rx1 = c_values[2] / (c_values[2] - c_values[3]);
                pos_x1  = quad[2] + (quad[3] - quad[2]) * int_rx1;
              }
            if (c_values[0] * c_values[2] <= 0)
              {
                int_ry0 = c_values[0] / (c_values[0] - c_values[2]);
                pos_y0  = quad[0] + (quad[2] - quad[0]) * int_ry0;
              }
            if (c_values[1] * c_values[3] <= 0)
              {
                int_ry1 = c_values[1] / (c_values[1] - c_values[3]);
                pos_y1  = quad[1] + (quad[3] - quad[1]) * int_ry1;
              }
            Tensor<1, dim> difference;
            Tensor<2, dim> interface_p;
            if (int_rx0 > 0)
              {
                if (int_ry0 > 0)
                  {
                    const double my_area = 0.5 * int_rx0 * int_ry0;
                    local_area -= (c_values[0] < 0) ? my_area : 1 - my_area;
                    difference = pos_x0 - pos_y0;
                    perimeter += difference.norm();
                    interface_p[0] = pos_x0;
                    interface_p[1] = pos_y0;
                  //  pcout << "ry0>0 	" << std::endl;
                  //  pcout << "my_area = " << my_area << " difference = " << difference <<"  pos_x0 = " << pos_x0 <<"  pos_y0 = " << pos_y0 << std::endl;
                  }
                if (int_ry1 > 0)
                  {
                    const double my_area = 0.5 * (1 - int_rx0) * int_ry1;
                    local_area -= (c_values[1] < 0) ? my_area : 1 - my_area;
                    difference = pos_x0 - pos_y1;
                    perimeter += difference.norm();
                    interface_p[0] = pos_x0;
                    interface_p[1] = pos_y1;
                  //  pcout << "ry1>0 	" << std::endl;
                  //  pcout << "my_area = " << my_area << " difference = " << difference <<"  pos_x0 = " << pos_x0 <<"  pos_y1 = " << pos_y1 << std::endl;
                  }
                if (int_rx1 > 0 && int_ry0 < 0 && int_ry1 < 0)
                  {
                    const double my_area = 0.5 * (int_rx0 + int_rx1);
                    local_area -= (c_values[0] < 0) ? my_area : 1 - my_area;
                    difference = pos_x0 - pos_x1;
                    perimeter += difference.norm();
                    interface_p[0] = pos_x0;
                    interface_p[1] = pos_x1;
                  //  pcout << "rx1>0 	" << std::endl;
                  //  pcout << "my_area = " << my_area << " difference = " << difference <<"  pos_x0 = " << pos_x0 <<"  pos_x1 = " << pos_x1 << std::endl;
                  }
              }
            if (int_rx1 > 0)
              {
                if (int_ry0 > 0)
                  {
                    const double my_area = 0.5 * int_rx1 * (1 - int_ry0);
                    local_area -= (c_values[2] < 0) ? my_area : 1 - my_area;
                    difference = pos_x1 - pos_y0;
                    perimeter += difference.norm();
                    interface_p[0] = pos_x1;
                    interface_p[1] = pos_y0;
                   // pcout << "rx1>0 and ry0>0 	" << std::endl;
                    //pcout << "my_area = " << my_area << " difference = " << difference <<"  pos_x1 = " << pos_x1 <<"  pos_y0 = " << pos_y0 << std::endl;
                  }
                if (int_ry1 > 0)
                  {
                    const double my_area = 0.5 * (1 - int_rx1) * (1 - int_ry1);
                    local_area -= (c_values[3] < 0) ? my_area : 1 - my_area;
                    difference = pos_x1 - pos_y1;
                    perimeter += difference.norm();
                    interface_p[0] = pos_x1;
                    interface_p[1] = pos_y1;
                   // pcout << "rx1>0 and ry1>0 	" << std::endl;
                   // pcout << "my_area = " << my_area << " difference = " << difference <<"  pos_x1 = " << pos_x1 <<"  pos_y1 = " << pos_y1 << std::endl;
                  }
              }
            if (int_ry0 > 0 && int_ry1 > 0 && int_rx0 < 0 && int_rx1 < 0)
              {
                const double my_area = 0.5 * (int_ry0 + int_ry1);
                local_area -= (c_values[0] < 0) ? my_area : 1 - my_area;
                difference = pos_y0 - pos_y1;
                perimeter += difference.norm();
                interface_p[0] = pos_y0;
                interface_p[1] = pos_y1;
                //pcout << "rx1>0 and rest 	" << std::endl;
                //pcout << "my_area = " << my_area << " difference = " << difference <<"  pos_y0 = " << pos_y0 <<"  pos_y1 = " << pos_y1 << std::endl;
              }
            if (int_rx0 <= 0 && int_rx1 <= 0 && int_ry0 <= 0 && int_ry1 <= 0 &&
                c_values[0] <= 0)
              local_area = 0;

            if (interface_p != Tensor<2, dim>() && interface_points != 0)
              interface_points->push_back(interface_p);

            Assert(local_area >= 0, ExcMessage("Substracted too much"));
            for (unsigned int i = 0; i < n_points; ++i)
              {
                double my_area = local_area * quad_weights[i];
                //pcout << "local_area = " << local_area << "   quad weights= "<< quad_weights[i] << std::endl;
                //pcout << "my_area = " << my_area << std::endl;
                area += my_area;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    center_of_mass[d] += quad[i][d] * my_area;
                    velocity[d] += velocities[i][d] * my_area;
                    //pcout << "quad = " << quad[i][d] << std::endl;
                  }
              }
          }
      }

  const MPI_Comm &mpi_communicator = get_communicator(triangulation);

  const double global_area      = Utilities::MPI::sum(area, mpi_communicator);
  const double global_perimeter = Utilities::MPI::sum(perimeter, mpi_communicator);
  pcout << "area = " << area << "   perimeter = " << perimeter << std::endl;

  Tensor<1, dim> global_mass_center;
  Tensor<1, dim> global_velocity;

  for (unsigned int d = 0; d < dim; ++d)
    {
      global_velocity[d]    = Utilities::MPI::sum(velocity[d], mpi_communicator);
      global_mass_center[d] = Utilities::MPI::sum(center_of_mass[d], mpi_communicator);
      pcout << "d = " << d << " :   center of mass = " << center_of_mass[d] <<  std::endl;
    }
  
  two_phase_solver->set_adaptive_time_step(global_velocity.norm() / global_area);

  const double circularity = 2. * std::sqrt(global_area * numbers::PI) / global_perimeter;
  if (parameters.output_verbosity > 0)
    {
      const std::size_t old_precision = std::cout.precision();
      std::cout.precision(8);
      pcout << "  Degree of circularity: " << circularity << std::endl;
      pcout << "  Mean bubble velocity: ";
      for (unsigned int d = 0; d < dim; ++d)
        pcout << ((std::abs(global_velocity[d]) < 1e-7 * global_velocity.norm()) ?
                    0. :
                    (global_velocity[d] / global_area))
              << "  ";
      pcout << std::endl;
      pcout << "  Position of the center of mass:  ";
      for (unsigned int d = 0; d < dim; ++d)
        pcout << ((std::abs(global_mass_center[d]) < 1e-7 * global_omega_diameter) ?
                    0. :
                    (global_mass_center[d] / global_area))
              << "  ";
      pcout << std::endl;
      pcout << "with diameter = " << global_omega_diameter << std::endl;

 /*     std::pair<double, double> concentration = get_concentration_range(level_set_solver);
      pcout << "  Range of level set values: " << concentration.first << " / "
            << concentration.second << std::endl;
*/
/*      std::cout.precision(old_precision);
    }

  std::vector<double> data(4 + 2 * dim);
  data[0] = navier_stokes_solver.time_stepping.now();
  data[1] = global_area;
  data[2] = global_perimeter;
  data[3] = circularity;
  for (unsigned int d = 0; d < dim; ++d)
    data[4 + d] = global_velocity[d] / global_area;
  for (unsigned int d = 0; d < dim; ++d)
    data[4 + dim + d] = global_mass_center[d] / global_area;

  // get interface points from other processors
  if (interface_points != 0)
    {
      std::vector<unsigned int> receive_count(
        Utilities::MPI::n_mpi_processes(mpi_communicator));

      unsigned int n_send_elements = interface_points->size();

      MPI_Gather(&n_send_elements,
                 1,
                 MPI_UNSIGNED,
                 &receive_count[0],
                 1,
                 MPI_UNSIGNED,
                 0,
                 mpi_communicator);
      for (unsigned int i = 1; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
        {
          // Each processor sends the interface_points he deals with to
          // processor
          // 0
          if (Utilities::MPI::this_mpi_process(mpi_communicator) == i)
            {
              // put data into a std::vector<double> to create a data type that
              // MPI understands
              std::vector<double> send_data(2 * dim * interface_points->size());
              for (unsigned int j = 0; j < interface_points->size(); ++j)
                for (unsigned int d = 0; d < 2; ++d)
                  for (unsigned int e = 0; e < dim; ++e)
                    send_data[j * 2 * dim + d * dim + e] = (*interface_points)[j][d][e];
              MPI_Send(
                &send_data[0], send_data.size(), MPI_DOUBLE, 0, i, mpi_communicator);

              // when we are done with sending, destroy the data on all
              // processors except processor 0
              std::vector<Tensor<2, dim>> empty;
              interface_points->swap(empty);
            }

          // Processor 0 receives data from the other processors
          if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            {
              std::vector<double> receive_data(2 * dim * receive_count[i]);
              int                 ierr = MPI_Recv(&receive_data[0],
                                  receive_data.size(),
                                  MPI_DOUBLE,
                                  i,
                                  i,
                                  mpi_communicator,
                                  MPI_STATUSES_IGNORE);
              (void)ierr;
              Assert(ierr == MPI_SUCCESS, ExcInternalError());
              for (unsigned int j = 0; j < receive_count[i]; ++j)
                {
                  Tensor<2, dim> point;
                  for (unsigned int d = 0; d < 2; ++d)
                    for (unsigned int e = 0; e < dim; ++e)
                      point[d][e] = receive_data[j * 2 * dim + d * dim + e];
                  interface_points->push_back(point);
                }
            }
        }
    }

    pcout << "reach end" <<std::endl;

  return data;
}

*/


template <int dim>
void
MicroFluidicProblem<dim>::run()
{
  std::vector<unsigned int> subdivisions(dim, 5);
  subdivisions[dim - 1] = 10;

  const Point<dim> bottom_left;
  const Point<dim> top_right = (dim == 2 ? Point<dim>(1, 2) : Point<dim>(1, 1, 2));
  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            subdivisions,
                                            bottom_left,
                                            top_right);

  Triangulation<dim - 1, dim> surface_mesh;
  GridGenerator::hyper_sphere(surface_mesh, Point<dim>(0.5, 0.5), 0.25);
  surface_mesh.refine_global(5);

  typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin(),
    endc = triangulation.end();

  for (; cell != endc; ++cell)
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      if (cell->face(face)->at_boundary() &&
          (std::fabs(cell->face(face)->center()[0] - 1) < 1e-14 ||
           std::fabs(cell->face(face)->center()[0]) < 1e-14))
        cell->face(face)->set_boundary_id(2);

  AssertThrow(parameters.global_refinements < 12, ExcInternalError());
  
  global_omega_diameter = GridTools::diameter(triangulation);
  pcout << "after diameter = " << global_omega_diameter << std::endl;

  NavierStokes<dim> navier_stokes_solver(parameters, triangulation, &timer);
  

  navier_stokes_solver.set_no_slip_boundary(0);
  navier_stokes_solver.fix_pressure_constant(0);
  navier_stokes_solver.set_symmetry_boundary(2);
  // navier_stokes_solver.boundary->fluid_type[0] =
  // std::make_shared<Functions::ConstantFunction<dim>>(1.0);

  navier_stokes_solver.setup_problem(Functions::ZeroFunction<dim>(dim));
  navier_stokes_solver.print_n_dofs();

  std::unique_ptr<SharpInterfaceSolver> solver;
 
  if (parameters.solver_method == "front tracking")
    solver =
      std::make_unique<FrontTrackingSolver<dim>>(navier_stokes_solver, surface_mesh);
  else if (parameters.solver_method == "mixed level set")
    solver = std::make_unique<MixedLevelSetSolver<dim>>(navier_stokes_solver,
                                                        surface_mesh,
                                                        InitialValuesLS<dim>());
  else if (parameters.solver_method == "sharp level set")
    solver = std::make_unique<MixedLevelSetSolver<dim>>(navier_stokes_solver,
                                                        InitialValuesLS<dim>());
  else if (parameters.solver_method == "level set")
    solver = std::make_unique<MixedLevelSetSolver<dim>>(navier_stokes_solver,
                                                        InitialValuesLS<dim>(),
                                                        false);
  else
    AssertThrow(false, ExcNotImplemented());

  solver->output_solution(parameters.output_filename);
  // bubble statistics
  solution_data.push_back(solver->compute_bubble_statistics(global_omega_diameter,0));
  
  bool first_output = true;
  while (navier_stokes_solver.time_stepping.at_end() == false)
    {
      solver->advance_time_step();

      solver->output_solution(parameters.output_filename);

      // evaluate velocity norm and pressure jump
      evaluate_spurious_velocities(navier_stokes_solver);
      // evaluate bubble
      solution_data.push_back(solver->compute_bubble_statistics(global_omega_diameter,0));

      if (solution_data.size() > 0 &&
        Utilities::MPI::this_mpi_process(triangulation.get_communicator()) == 0 &&
        //two_phase_solver->get_time_stepping().at_tick(parameters.output_frequency));
        navier_stokes_solver.time_stepping.at_tick(parameters.output_frequency))
      {
        const int time_step = 1.000001e4 * navier_stokes_solver.time_stepping.step_size();
        std::ostringstream filename3;
        filename3 << parameters.output_filename << "-"
                  << Utilities::int_to_string((int)parameters.adaptive_refinements, 1)
                  << "-" << Utilities::int_to_string(parameters.global_refinements, 3)
                  << "-" << Utilities::int_to_string(time_step, 4) << ".txt";

        std::fstream output_positions3(filename3.str().c_str(),
                                        first_output ? std::ios::out :
                                                      std::ios::out | std::ios::app);

        output_positions3.precision(14);
        if (first_output)
          output_positions3
            << "#    time        area      perimeter   circularity   bubble_xvel   bubble_yvel   bubble_xpos    bubble_ypos"
            << std::endl;
        for (unsigned int i = 0; i < solution_data.size(); ++i)
          {
            output_positions3 << " ";
            for (unsigned int j = 0; j < solution_data[i].size(); ++j)
              output_positions3 << solution_data[i][j] << "   ";
            output_positions3 << std::endl;
          }
        solution_data.clear();
        first_output = false;
      }

    }
}

int
main(int argc, char **argv)
{
  using namespace dealii;


  try
    {
      deallog.depth_console(0);
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

      std::string paramfile;
      if (argc > 1)
        paramfile = argv[1];
      else
        paramfile = "sharp_interfaces_04.prm";

      TwoPhaseParameters parameters(paramfile);
      if (parameters.dimension == 2)
        {
          MicroFluidicProblem<2> flow_problem(parameters);
          flow_problem.run();
        }
      else
        AssertThrow(false, ExcNotImplemented());
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;
}
