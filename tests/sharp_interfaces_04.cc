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

#include <adaflo/level_set_okz.h>
#include <adaflo/level_set_okz_matrix.h>
#include <adaflo/parameters.h>
#include <adaflo/phase_field.h>

#include "sharp_interfaces_util.h"


using namespace dealii;

using VectorType = LinearAlgebra::distributed::Vector<double>;


struct TwoPhaseParameters : public FlowParameters
{
  TwoPhaseParameters(const std::string &parameter_filename)
  {
    ParameterHandler prm;
    FlowParameters::declare_parameters(prm);
    prm.enter_subsection("Problem-specific");
    prm.declare_entry("two-phase method",
                      "level set okz",
                      Patterns::Selection(
                        "level set okz|level set okz matrix|phase field"),
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
class MicroFluidicProblem
{
public:
  MicroFluidicProblem(const TwoPhaseParameters &parameters);
  void
  run();

private:
  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;

  mutable TimerOutput timer;

  TwoPhaseParameters                        parameters;
  parallel::distributed::Triangulation<dim> triangulation;
};


template <int dim>
MicroFluidicProblem<dim>::MicroFluidicProblem(const TwoPhaseParameters &parameters)
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , timer(pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  , parameters(parameters)
  , triangulation(mpi_communicator)
{}



template <int dim>
class SharpInterfaceSolver
{
public:
  SharpInterfaceSolver(NavierStokes<dim> &          navier_stokes_solver,
                       Triangulation<dim - 1, dim> &surface_mesh)
    : navier_stokes_solver(navier_stokes_solver)
    , euler_dofhandler(surface_mesh)
    , euler_mapping(euler_dofhandler, euler_vector)
  {
    const unsigned int fe_degree      = 3;
    const unsigned int mapping_degree = fe_degree;

    FESystem<dim - 1, dim> euler_fe(FE_Q<dim - 1, dim>(fe_degree), dim);
    euler_dofhandler.distribute_dofs(euler_fe);

    euler_vector(euler_dofhandler.n_dofs());
    VectorTools::get_position_vector(euler_dofhandler,
                                     euler_vector,
                                     MappingQGeneric<dim - 1, dim>(mapping_degree));
  }

  void
  advance_time_step()
  {
    this->move_surface_mesh();
    this->update_phases();
    this->update_surface_tension();
    this->update_gravity_force();
    navier_stokes_solver.advance_time_step();
  }

  void
  output_solution(const std::string &output_filename)
  {
    navier_stokes_solver.output_solution(output_filename);
  }

private:
  void
  move_surface_mesh()
  {
    VectorTools::update_position_vector(navier_stokes_solver.time_stepping.step_size(),
                                        navier_stokes_solver.get_dof_handler_u(),
                                        navier_stokes_solver.mapping,
                                        navier_stokes_solver.solution_update.block(0),
                                        euler_dofhandler,
                                        euler_mapping,
                                        euler_vector);
  }

  void
  update_phases()
  {}

  void
  update_surface_tension()
  {}

  void
  update_gravity_force()
  {}

  // background mesh
  NavierStokes<dim> &navier_stokes_solver;

  // surface mesh
  DoFHandler<dim - 1, dim>                       euler_dofhandler;
  VectorType                                     euler_vector;
  const MappingFEField<dim - 1, dim, VectorType> euler_mapping;
};



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

  NavierStokes<dim> navier_stokes_solver(parameters, triangulation, &timer);

  navier_stokes_solver.set_no_slip_boundary(0);
  navier_stokes_solver.fix_pressure_constant(0);
  navier_stokes_solver.set_symmetry_boundary(2);

  navier_stokes_solver.setup_problem(Functions::ZeroFunction<dim>(dim));

  Triangulation<dim - 1, dim> surface_mesh;

  SharpInterfaceSolver<dim> solver(navier_stokes_solver, surface_mesh);

  solver.output_solution(parameters.output_filename);

  while (navier_stokes_solver.time_stepping.at_end() == false)
    {
      solver.advance_time_step();

      solver.output_solution(parameters.output_filename);
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
        paramfile = "rising_bubble.prm";

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
