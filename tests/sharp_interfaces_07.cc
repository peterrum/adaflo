#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

template <int dim>
class LSFunction : public Function<dim>
{
public:
  LSFunction()
    : Function<dim>(1, 0)
  {}

  double
  value(const Point<dim> &p, const unsigned int) const
  {
    return p.distance(Point<dim>()) - 0.5;
  }
};

template <int dim, int spacedim>
void
process_sub_cell(const std::vector<double> &         ls_values,
                 const std::vector<Point<spacedim>> &points,
                 const std::vector<unsigned int>     mask,
                 std::vector<Point<spacedim>> &      vertices,
                 std::vector<::CellData<dim>> &      cells)
{
  unsigned int c = 0;

  for (unsigned int i = 0, scale = 1; i < 4; ++i, scale *= 2)
    c += (ls_values[mask[i]] > 0) * scale;

  if (c == 0 || c == 15)
    return;


  const auto process_points = [&](const auto &lines) {
    const double w0 = std::abs(ls_values[mask[lines[0]]]);
    const double w1 = std::abs(ls_values[mask[lines[1]]]);

    return points[mask[lines[0]]] * (w1 / (w0 + w1)) +
           points[mask[lines[1]]] * (w0 / (w0 + w1));
  };

  const auto process_lines = [&](const auto &lines) {
    std::array<std::array<unsigned int, 2>, 4> table{
      {{{0, 3}}, {{1, 2}}, {{0, 1}}, {{3, 2}}}};

    const auto p0 = process_points(table[lines[0]]);
    const auto p1 = process_points(table[lines[1]]);

    cells.resize(cells.size() + 1);
    cells.back().vertices[0] = vertices.size();
    cells.back().vertices[1] = vertices.size() + 1;

    vertices.emplace_back(p0);
    vertices.emplace_back(p1);
  };

  if (c == 5 || c == 10)
    {
      Assert(false, ExcNotImplemented());
      return;
    }

  static const unsigned int X = -1;

  std::array<std::array<unsigned int, 2>, 16> table{{
    {{X, X}},
    {{0, 2}},
    {{1, 2}},
    {{0, 1}}, //  0- 3
    {{1, 3}},
    {{X, X}},
    {{2, 3}},
    {{0, 3}}, //  4- 7
    {{0, 3}},
    {{2, 3}},
    {{X, X}},
    {{1, 3}}, //  8-11
    {{0, 1}},
    {{2, 1}},
    {{0, 2}},
    {{X, X}} // 12-15
  }};

  process_lines(table[c]);

  std::cout << vertices.size() << " " << cells.size() << std::endl;
}

template <int dim>
void
test()
{
  const unsigned int n_refinements = 2;
  const unsigned int fe_degree     = 2;

  std::vector<Point<dim>> quadrature_points;
  quadrature_points.emplace_back(0, 0);
  quadrature_points.emplace_back(1, 0);
  quadrature_points.emplace_back(0, 1);
  quadrature_points.emplace_back(1, 1);
  quadrature_points.emplace_back(0.5, 0.5);

  Quadrature<dim> quadrature(quadrature_points);

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>{fe_degree});

  Vector<double>      ls_vector(dof_handler.n_dofs());
  std::vector<double> ls_values(quadrature.size());

  MappingQGeneric<dim> mapping(1);

  VectorTools::interpolate(mapping, dof_handler, LSFunction<dim>(), ls_vector);

  FEValues<dim> fe_values(mapping,
                          dof_handler.get_fe(),
                          quadrature,
                          update_values | update_quadrature_points);

  std::vector<Point<dim>>          vertices;
  std::vector<::CellData<dim - 1>> cells;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      fe_values.get_function_values(ls_vector, ls_values);

      std::vector<unsigned int> mask{0, 1, 3, 2, 4};

      process_sub_cell(
        ls_values, fe_values.get_quadrature_points(), mask, vertices, cells);
    }
  Triangulation<dim - 1, dim> tria_interface;
  tria_interface.create_triangulation(vertices, cells, {});

  {
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;

    DataOut<dim> data_out;
    data_out.set_flags(flags);

    data_out.add_data_vector(dof_handler, ls_vector, "ls");
    data_out.build_patches(mapping, 4);

    std::ofstream out("sharp_interfaces_07_a.vtk");
    data_out.write_vtk(out);
  }

  {
    std::ofstream out("sharp_interfaces_07_b.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria_interface, out);
  }
}

int
main()
{
  test<2>();
}